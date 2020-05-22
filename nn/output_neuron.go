package nn

import (
	"github.com/levinholsety/common-go/num"
)

func NewOutputNeuron(name string, bias num.Scalar, actFuncName string, errFuncName string) *OutputNeuron {
	neuron := &OutputNeuron{
		NeuronBase:  newNeuronBase(name, bias, actFuncName),
		errFuncName: errFuncName,
		errFunc:     NewErrorFunction(errFuncName),
	}
	return neuron
}

type OutputNeuron struct {
	*NeuronBase
	errFuncName string
	errFunc     ErrorFunction
	targetValue num.Tensor
	errorValue  num.Tensor
}

func (p *OutputNeuron) ready() {
	p.inputChannel = make(chan num.Tensor, len(p.srcNeurons))
}

func (p *OutputNeuron) activate(net *NeuralNetwork) {
	p.calcOutputValue()
	if net.learning {
		p.errorValue = p.errFunc.Func(p.targetValue, p.outputValue)
		p.deltaValue = p.errFunc.PDFunc(p.targetValue, p.outputValue).Mul(p.actFunc.PDFunc(p.outputValue))
		p.backward(net)
	}
}

func (p *OutputNeuron) SetTargetValue(v num.Tensor) {
	p.targetValue = v
}
