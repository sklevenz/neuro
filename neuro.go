package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"time"
)

// Sigmoid function and its derivative
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

// Random weight initialization
func randomWeight(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.Float64()*2 - 1 // random float between -1 and 1
	}
	return mat.NewDense(rows, cols, data)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Training data: 1x1 multiplication table
	inputData := mat.NewDense(9, 2, []float64{
		1, 1,
		1, 2,
		1, 3,
		2, 1,
		2, 2,
		2, 3,
		3, 1,
		3, 2,
		3, 3,
	})
	expectedOutput := mat.NewDense(9, 1, []float64{
		1,
		2,
		3,
		2,
		4,
		6,
		3,
		6,
		9,
	})

	// Initialize neural network with random weights
	inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons := 2, 3, 1
	hiddenLayerWeights := randomWeight(hiddenLayerNeurons, inputLayerNeurons)
	outputLayerWeights := randomWeight(outputLayerNeurons, hiddenLayerNeurons)

	learningRate := 0.5
	epochs := 10000

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		// Forward pass
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(inputData, hiddenLayerWeights.T())
		hiddenLayerActivations := new(mat.Dense)
		hiddenLayerActivations.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, outputLayerWeights.T())
		output := new(mat.Dense)
		output.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, outputLayerInput)

		// Calculate error
		error := new(mat.Dense)
		error.Sub(expectedOutput, output)
		dOutput := new(mat.Dense)
		dOutput.Apply(func(_, _ int, v float64) float64 { return sigmoidDerivative(v) }, output)
		errorOutputLayer := new(mat.Dense)
		errorOutputLayer.MulElem(error, dOutput)

		// Backpropagation
		hiddenLayerT := new(mat.Dense)
		hiddenLayerT.Mul(outputLayerWeights, errorOutputLayer.T())
		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.Apply(func(_, _ int, v float64) float64 { return sigmoidDerivative(v) }, hiddenLayerActivations)
		errorHiddenLayer := new(mat.Dense)
		errorHiddenLayer.MulElem(dHiddenLayer, hiddenLayerT)

		// Update weights
		outputLayerWeightsAdj := new(mat.Dense)
		outputLayerWeightsAdj.Mul(errorOutputLayer.T(), hiddenLayerActivations)
		outputLayerWeightsAdj.Scale(learningRate, outputLayerWeightsAdj)
		outputLayerWeights.Add(outputLayerWeights, outputLayerWeightsAdj.T())

		hiddenLayerWeightsAdj := new(mat.Dense)
		hiddenLayerWeightsAdj.Mul(errorHiddenLayer.T(), inputData)
		hiddenLayerWeightsAdj.Scale(learningRate, hiddenLayerWeightsAdj)
		hiddenLayerWeights.Add(hiddenLayerWeights, hiddenLayerWeightsAdj.T())
	}

	// Predictions after training
	fmt.Println("Predictions after training")
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(inputData, hiddenLayerWeights.T())
	hiddenLayerActivations := new(mat.Dense)
	hiddenLayerActivations.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, outputLayerWeights.T())
	output := new(mat.Dense)
	output.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, outputLayerInput)
	output.Apply(func(_, _ int, v float64) float64 { return math.Round(v*9) / 1 }, output)

	outputData := mat.Formatted(output, mat.Prefix("    "))
	fmt.Printf("output = %v\n", outputData)
}

