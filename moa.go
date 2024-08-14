// File: moa.go

package gollm

import (
	"context"
	"fmt"
	"sync"
)

// MOAConfig represents the configuration for the Mixture of Agents
type MOAConfig struct {
	Iterations int
	Models     [][]ConfigOption // Each inner slice represents a model's configuration
}

// MOALayer represents a single layer in the Mixture of Agents
type MOALayer struct {
	Models []LLM
}

// MOA represents the Mixture of Agents
type MOA struct {
	Config     MOAConfig
	Layers     []MOALayer
	Aggregator LLM
}

// NewMOA creates a new Mixture of Agents instance
func NewMOA(config MOAConfig, aggregatorConfig []ConfigOption) (*MOA, error) {
	if len(config.Models) == 0 {
		return nil, fmt.Errorf("invalid model configuration: at least one model must be specified")
	}

	moa := &MOA{
		Config: config,
		Layers: make([]MOALayer, len(config.Models)),
	}

	for i, modelConfig := range config.Models {
		llm, err := NewLLM(modelConfig...)
		if err != nil {
			return nil, fmt.Errorf("failed to create LLM for model %d: %w", i, err)
		}
		moa.Layers[i] = MOALayer{
			Models: []LLM{llm},
		}
	}

	aggregator, err := NewLLM(aggregatorConfig...)
	if err != nil {
		return nil, fmt.Errorf("failed to create aggregator LLM: %w", err)
	}
	moa.Aggregator = aggregator

	return moa, nil
}

// Generate processes the input through the Mixture of Agents and returns the final output
func (moa *MOA) Generate(ctx context.Context, input string) (string, error) {
	var layerOutputs []string

	for i := 0; i < moa.Config.Iterations; i++ {
		layerInput := input
		for _, layer := range moa.Layers {
			layerOutput, err := moa.processLayer(ctx, layer, layerInput)
			if err != nil {
				return "", fmt.Errorf("error processing layer: %w", err)
			}
			layerInput = layerOutput
		}
		layerOutputs = append(layerOutputs, layerInput)
	}

	return moa.aggregate(ctx, layerOutputs)
}

func (moa *MOA) processLayer(ctx context.Context, layer MOALayer, input string) (string, error) {
	var wg sync.WaitGroup
	results := make([]string, len(layer.Models))
	errors := make([]error, len(layer.Models))

	for i, model := range layer.Models {
		wg.Add(1)
		go func(index int, llm LLM) {
			defer wg.Done()
			output, err := llm.Generate(ctx, NewPrompt(input))
			if err != nil {
				errors[index] = err
				return
			}
			results[index] = output
		}(i, model)
	}

	wg.Wait()

	for _, err := range errors {
		if err != nil {
			return "", fmt.Errorf("error in layer processing: %w", err)
		}
	}

	return moa.combineResults(results), nil
}

func (moa *MOA) combineResults(results []string) string {
	combined := ""
	for _, result := range results {
		combined += result + "\n---\n"
	}
	return combined
}

func (moa *MOA) aggregate(ctx context.Context, outputs []string) (string, error) {
	aggregationPrompt := fmt.Sprintf("Synthesise these responses into a single, high-quality response:\n\n%s", moa.combineResults(outputs))
	return moa.Aggregator.Generate(ctx, NewPrompt(aggregationPrompt))
}
