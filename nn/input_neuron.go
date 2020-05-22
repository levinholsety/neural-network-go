package nn

import "github.com/levinholsety/common-go/num"

func NewInputNeuron(name string) *InputNeuron {
	return &InputNeuron{
		NeuronBase: newNeuronBase(name, 0, ""),
	}
}

type InputNeuron struct {
	*NeuronBase
}

func (p *InputNeuron) ready() {
	p.deltaChannel = make(chan num.Tensor, len(p.dstNeurons))
}

func (p *InputNeuron) activate(net *NeuralNetwork) {
	p.forward(net)
	if net.learning {
		p.calcDeltaValue()
	}
}

func (p *InputNeuron) SetInputValue(v num.Tensor) {
	p.outputValue = v
}
