package nn

import "github.com/levinholsety/common-go/num"

type Neuron interface {
	ID() int
	Name() string
	OutputValue() num.Tensor
	addSourceNeuron(neuron Neuron)
	addDestinationNeuron(neuron Neuron)
	sendOutputValue(v num.Tensor)
	sendDeltaValue(v num.Tensor)
	ready()
	activate(net *NeuralNetwork)
}
