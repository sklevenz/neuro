package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"gonum.org/v1/gonum/mat"
)

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func randomWeights(rows, cols int) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rand.Float64()*2 - 1 // random float between -1 and 1
	}
	return mat.NewDense(rows, cols, data)
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// 1x1 multiplication table (1 to 3)
	inputs := mat.NewDense(9, 2, []float64{
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
	expected := mat.NewDense(9, 1, []float64{
		1, 2, 3,
		2, 4, 6,
		3, 6, 9,
	})

	// Neural network dimensions
	inputLayer, hiddenLayer, outputLayer := 2, 3, 1

	// Initialize random weights
	hiddenWeights := randomWeights(hiddenLayer, inputLayer)
	outputWeights := randomWeights(outputLayer, hiddenLayer)

	// Hyperparameters
	learningRate := 0.5
	epochs := 20000

	for epoch := 0; epoch < epochs; epoch++ {
		// Forward Pass
		hiddenInput := new(mat.Dense)
		hiddenInput.Mul(inputs, hiddenWeights.T())
		hiddenOutput := new(mat.Dense)
		hiddenOutput.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, hiddenInput)

		outputInput := new(mat.Dense)
		outputInput.Mul(hiddenOutput, outputWeights.T())
		output := new(mat.Dense)
		output.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, outputInput)

		// Calculate the error
		error := new(mat.Dense)
		error.Sub(expected, output)

		// Backpropagation
		dOutput := new(mat.Dense)
		dOutput.Apply(func(_, _ int, v float64) float64 { return sigmoidDerivative(v) }, output)
		outputError := new(mat.Dense)
		outputError.MulElem(error, dOutput)

		hiddenLayerT := new(mat.Dense)
		hiddenLayerT.Mul(outputWeights, outputError.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.Apply(func(_, _ int, v float64) float64 { return sigmoidDerivative(v) }, hiddenOutput)
		hiddenError := new(mat.Dense)
		hiddenError.MulElem(dHiddenLayer, hiddenLayerT.T())

		// Update the weights
		hiddenErrorT := new(mat.Dense)
		hiddenErrorT.Mul(hiddenError.T(), inputs)
		hiddenErrorT.Scale(learningRate, hiddenErrorT)
		hiddenWeights.Add(hiddenWeights, hiddenErrorT.T())

		outputErrorT := new(mat.Dense)
		outputErrorT.Mul(outputError.T(), hiddenOutput)
		outputErrorT.Scale(learningRate, outputErrorT)
		outputWeights.Add(outputWeights, outputErrorT.T())
	}

	// Predictions after training
	hiddenInput := new(mat.Dense)
	hiddenInput.Mul(inputs, hiddenWeights.T())
	hiddenOutput := new(mat.Dense)
	hiddenOutput.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, hiddenInput)

	outputInput := new(mat.Dense)
	outputInput.Mul(hiddenOutput, outputWeights.T())
	output := new(mat.Dense)
	output.Apply(func(_, _ int, v float64) float64 { return sigmoid(v) }, outputInput)

	output.Apply(func(_, _ int, v float64) float64 { return math.Round(v * 10) }, output)

	fmt.Println("Predictions after training:")
	fmt.Println(mat.Formatted(output, mat.Prefix("    ")))
}
