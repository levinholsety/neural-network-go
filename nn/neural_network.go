package nn

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"sync"

	"github.com/levinholsety/common-go/num"
	"github.com/levinholsety/neural-network/nn/model"
)

func NewNeuralNetwork() *NeuralNetwork {
	return &NeuralNetwork{
		neurons:     map[int]Neuron{},
		connections: map[[2]int]*Connection{},
	}
}

type NeuralNetwork struct {
	TotalError    float64
	inputNeurons  []*InputNeuron
	outputNeurons []*OutputNeuron
	hiddenNeurons []*HiddenNeuron
	neurons       map[int]Neuron
	connections   map[[2]int]*Connection
	learning      bool
	learningRate  float64
	momentum      float64
	patternCount  int
}

func (p *NeuralNetwork) SetLearning(learning bool) {
	p.learning = learning
}

func (p *NeuralNetwork) SetLearningRate(learningRate float64) {
	p.learningRate = learningRate
}

func (p *NeuralNetwork) SetMomentum(momentum float64) {
	p.momentum = momentum
}

func (p *NeuralNetwork) ConnectNeurons(src, dst Neuron, weight num.Scalar) {
	src.addDestinationNeuron(dst)
	dst.addSourceNeuron(src)
	if _, ok := p.neurons[src.ID()]; !ok {
		p.neurons[src.ID()] = src
		if neuron, ok := src.(*InputNeuron); ok {
			p.inputNeurons = append(p.inputNeurons, neuron)
		} else if neuron, ok := src.(*HiddenNeuron); ok {
			p.hiddenNeurons = append(p.hiddenNeurons, neuron)
		}
	}
	if _, ok := p.neurons[dst.ID()]; !ok {
		p.neurons[dst.ID()] = dst
		if neuron, ok := dst.(*OutputNeuron); ok {
			p.outputNeurons = append(p.outputNeurons, neuron)
		} else if neuron, ok := dst.(*HiddenNeuron); ok {
			p.hiddenNeurons = append(p.hiddenNeurons, neuron)
		}
	}
	p.connections[[2]int{src.ID(), dst.ID()}] = &Connection{weight}
}

func (p *NeuralNetwork) ConnectLayers(src, dst Layer, weights num.Matrix) {
	for i, srcNeuron := range src {
		for j, dstNeuron := range dst {
			p.ConnectNeurons(srcNeuron, dstNeuron, weights[i][j])
		}
	}
}

func (p *NeuralNetwork) SetPatterns(patterns num.Matrix) {
	p.patternCount = patterns.Size().RowCount
	var m num.Matrix = patterns.T().(num.Matrix)
	for i, row := range m[:len(p.inputNeurons)] {
		p.inputNeurons[i].SetInputValue(row)
	}
	for i, row := range m[len(p.inputNeurons):] {
		p.outputNeurons[i].SetTargetValue(row)
	}
}

func (p *NeuralNetwork) Go() {
	for _, neuron := range p.neurons {
		neuron.ready()
	}
	var wg sync.WaitGroup
	for _, neuron := range p.neurons {
		wg.Add(1)
		go func(neuron Neuron) {
			neuron.activate(p)
			wg.Done()
		}(neuron)
	}
	wg.Wait()
	if p.learning {
		p.TotalError = 0
		for _, neuron := range p.outputNeurons {
			te := neuron.errorValue.Sum()
			p.TotalError += float64(te)
		}
		p.TotalError /= float64(len(p.outputNeurons))
	}
}

func (p *NeuralNetwork) PrintOutputValues() {
	for _, neuron := range p.outputNeurons {
		fmt.Printf("%s: %.10f\n", neuron.name, neuron.OutputValue())
	}
}

func (p *NeuralNetwork) Save(w io.Writer) (err error) {
	mnn := &model.NeuralNetwork{
		LearningRate: p.learningRate,
		Momentum:     p.momentum,
	}
	for _, neuron := range p.inputNeurons {
		mnn.InputNeurons = append(mnn.InputNeurons, &model.Neuron{
			ID:   neuron.id,
			Name: neuron.name,
		})
	}
	for _, neuron := range p.outputNeurons {
		mnn.OutputNeurons = append(mnn.OutputNeurons, &model.Neuron{
			ID:          neuron.id,
			Name:        neuron.name,
			Bias:        float64(neuron.bias),
			ActFuncName: neuron.actFuncName,
			ErrFuncName: neuron.errFuncName,
		})
	}
	for _, neuron := range p.hiddenNeurons {
		mnn.HiddenNeurons = append(mnn.HiddenNeurons, &model.Neuron{
			ID:          neuron.id,
			Name:        neuron.name,
			Bias:        float64(neuron.bias),
			ActFuncName: neuron.actFuncName,
		})
	}
	for ids, conn := range p.connections {
		mnn.Connections = append(mnn.Connections, &model.Connection{
			SrcID:  ids[0],
			DstID:  ids[1],
			Weight: float64(conn.Weight),
		})
	}
	data, err := json.Marshal(mnn)
	if err != nil {
		return
	}
	_, err = w.Write(data)
	return
}

func (p *NeuralNetwork) Load(r io.Reader) (err error) {
	data, err := ioutil.ReadAll(r)
	if err != nil {
		return
	}
	mnn := new(model.NeuralNetwork)
	err = json.Unmarshal(data, mnn)
	if err != nil {
		return
	}
	p.learningRate = mnn.LearningRate
	p.momentum = mnn.Momentum
	for _, mn := range mnn.InputNeurons {
		neuron := &InputNeuron{
			NeuronBase: &NeuronBase{
				id:   mn.ID,
				name: mn.Name,
			},
		}
		p.inputNeurons = append(p.inputNeurons, neuron)
		p.neurons[mn.ID] = neuron
	}
	for _, mn := range mnn.OutputNeurons {
		neuron := &OutputNeuron{
			NeuronBase: &NeuronBase{
				id:          mn.ID,
				name:        mn.Name,
				bias:        num.Scalar(mn.Bias),
				actFuncName: mn.ActFuncName,
				actFunc:     NewActivationFunction(mn.ActFuncName),
			},
			errFuncName: mn.ErrFuncName,
			errFunc:     NewErrorFunction(mn.ErrFuncName),
		}
		p.outputNeurons = append(p.outputNeurons, neuron)
		p.neurons[mn.ID] = neuron
	}
	for _, mn := range mnn.HiddenNeurons {
		neuron := &HiddenNeuron{
			NeuronBase: &NeuronBase{
				id:          mn.ID,
				name:        mn.Name,
				bias:        num.Scalar(mn.Bias),
				actFuncName: mn.ActFuncName,
				actFunc:     NewActivationFunction(mn.ActFuncName),
			},
		}
		p.hiddenNeurons = append(p.hiddenNeurons, neuron)
		p.neurons[mn.ID] = neuron
	}
	for _, mc := range mnn.Connections {
		p.ConnectNeurons(p.neurons[mc.SrcID], p.neurons[mc.DstID], num.Scalar(mc.Weight))
	}
	return
}
