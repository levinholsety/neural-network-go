package nn

import (
	"fmt"

	"github.com/levinholsety/common-go/num"
)

func NewInputLayer(neuronCount int) Layer {
	neurons := make([]Neuron, neuronCount)
	for i := 0; i < neuronCount; i++ {
		neurons[i] = NewInputNeuron(fmt.Sprintf("in_%d", i))
	}
	return Layer(neurons)
}

func NewHiddenLayer(neuronCount, n int, bias num.Scalar, actFuncName string) Layer {
	neurons := make([]Neuron, neuronCount)
	for i := 0; i < neuronCount; i++ {
		neurons[i] = NewHiddenNeuron(fmt.Sprintf("hn_%d_%d", n, i), bias, actFuncName)
	}
	return Layer(neurons)
}

func NewOutputLayer(neuronCount int, bias num.Scalar, actFuncName, errFuncName string) Layer {
	neurons := make([]Neuron, neuronCount)
	for i := 0; i < neuronCount; i++ {
		neurons[i] = NewOutputNeuron(fmt.Sprintf("on_%d", i), bias, actFuncName, errFuncName)
	}
	return Layer(neurons)
}

type Layer []Neuron
