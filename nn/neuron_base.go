package nn

import (
	"github.com/levinholsety/common-go/comm"
	"github.com/levinholsety/common-go/num"
)

func newNeuronBase(name string, bias num.Scalar, actFuncName string) *NeuronBase {
	return &NeuronBase{
		id:          int(comm.GenerateID()),
		name:        name,
		bias:        bias,
		actFuncName: actFuncName,
		actFunc:     NewActivationFunction(actFuncName),
	}
}

type NeuronBase struct {
	id           int
	name         string
	bias         num.Scalar
	actFuncName  string
	actFunc      ActivationFunction
	srcNeurons   []Neuron
	dstNeurons   []Neuron
	inputChannel chan num.Tensor
	deltaChannel chan num.Tensor
	outputValue  num.Tensor
	deltaValue   num.Tensor
}

func (p *NeuronBase) ID() int {
	return p.id
}

func (p *NeuronBase) Name() string {
	return p.name
}

func (p *NeuronBase) OutputValue() num.Tensor {
	return p.outputValue.Duplicate()
}

func (p *NeuronBase) addSourceNeuron(neuron Neuron) {
	p.srcNeurons = append(p.srcNeurons, neuron)
}

func (p *NeuronBase) addDestinationNeuron(neuron Neuron) {
	p.dstNeurons = append(p.dstNeurons, neuron)
}

func (p *NeuronBase) sendOutputValue(v num.Tensor) {
	p.inputChannel <- v
}

func (p *NeuronBase) sendDeltaValue(v num.Tensor) {
	p.deltaChannel <- v
}

func (p *NeuronBase) calcOutputValue() {
	p.outputValue = num.Scalar(p.bias)
	for range p.srcNeurons {
		p.outputValue = p.outputValue.Add(<-p.inputChannel)
	}
	p.outputValue = p.actFunc.Func(p.outputValue)
}

func (p *NeuronBase) calcDeltaValue() {
	p.deltaValue = num.Scalar(0)
	for range p.dstNeurons {
		p.deltaValue = p.deltaValue.Add(<-p.deltaChannel)
	}
}

func (p *NeuronBase) forward(net *NeuralNetwork) {
	for _, neuron := range p.dstNeurons {
		conn := net.connections[[2]int{p.id, neuron.ID()}]
		neuron.sendOutputValue(p.outputValue.Mul(conn.Weight))
	}
}

func (p *NeuronBase) backward(net *NeuralNetwork) {
	biasDeltaValue := p.deltaValue.Mul(p.bias).Sum().Div(num.Scalar(net.patternCount))
	biasDeltaValue = num.Scalar(net.learningRate).Negative().Mul(biasDeltaValue).Add(num.Scalar(net.momentum).Mul(biasDeltaValue))
	p.bias = p.bias.Add(biasDeltaValue).(num.Scalar)
	for _, src := range p.srcNeurons {
		conn := net.connections[[2]int{src.ID(), p.id}]
		output := src.OutputValue()
		deltaValue := p.deltaValue.Mul(p.actFunc.PDFunc(output)).Mul(conn.Weight)
		src.sendDeltaValue(deltaValue)
		weightDeltaValue := p.deltaValue.Dot(output).Div(num.Scalar(net.patternCount))
		weightDeltaValue = num.Scalar(net.learningRate).Negative().Mul(weightDeltaValue).Add(num.Scalar(net.momentum).Mul(weightDeltaValue))
		conn.Weight = conn.Weight.Add(weightDeltaValue).(num.Scalar)
	}
}
