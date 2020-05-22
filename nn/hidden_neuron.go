package nn

import "github.com/levinholsety/common-go/num"

func NewHiddenNeuron(name string, bias num.Scalar, actFuncName string) *HiddenNeuron {
	neuron := &HiddenNeuron{
		NeuronBase: newNeuronBase(name, bias, actFuncName),
	}
	return neuron
}

type HiddenNeuron struct {
	*NeuronBase
}

func (p *HiddenNeuron) ready() {
	p.inputChannel = make(chan num.Tensor, len(p.srcNeurons))
	p.deltaChannel = make(chan num.Tensor, len(p.dstNeurons))
}

func (p *HiddenNeuron) activate(net *NeuralNetwork) {
	p.calcOutputValue()
	p.forward(net)
	if net.learning {
		p.calcDeltaValue()
		p.backward(net)
	}
}
