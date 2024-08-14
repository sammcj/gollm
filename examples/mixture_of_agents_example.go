package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/sammcj/gollm"
)

func main() {
	// Configure the MOA
	moaConfig := gollm.MOAConfig{
		Layers:     2,
		Models:     3,
		Iterations: 2,
	}

	// Configure individual models
	modelConfigs := [][]gollm.ConfigOption{
		{
			gollm.SetProvider("openai"), gollm.SetModel("gpt-3.5-turbo"), gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
			gollm.SetProvider("anthropic"), gollm.SetModel("claude-3-opus-20240229"), gollm.SetAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
			gollm.SetProvider("groq"), gollm.SetModel("llama-3.1-70b-versatile"), gollm.SetAPIKey(os.Getenv("GROQ_API_KEY")),
		},
		{
			gollm.SetProvider("openai"), gollm.SetModel("gpt-4o"), gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
			gollm.SetProvider("anthropic"), gollm.SetModel("claude-3-5-sonnet-20240620"), gollm.SetAPIKey(os.Getenv("ANTHROPIC_API_KEY")),
			gollm.SetProvider("openai"), gollm.SetModel("gpt-4o-mini"), gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
		},
	}

	// Configure the aggregator
	aggregatorConfig := []gollm.ConfigOption{
		gollm.SetProvider("openai"),
		gollm.SetModel("gpt-4o"),
		gollm.SetAPIKey(os.Getenv("OPENAI_API_KEY")),
	}

	// Create the MOA
	moa, err := gollm.NewMOA(moaConfig, modelConfigs, aggregatorConfig)
	if err != nil {
		log.Fatalf("Failed to create MOA: %v", err)
	}

	// Use the MOA to generate a response
	ctx := context.Background()
	input := "Explain the concept of quantum entanglement and its potential applications in computing."
	output, err := moa.Generate(ctx, input)
	if err != nil {
		log.Fatalf("Failed to generate response: %v", err)
	}

	fmt.Printf("Input: %s\n\n", input)
	fmt.Printf("MOA Response:\n%s\n", output)
}
