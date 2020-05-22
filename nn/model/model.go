package model

type NeuralNetwork struct {
	LearningRate  float64       `json:"learningRate"`
	Momentum      float64       `json:"momentum"`
	InputNeurons  []*Neuron     `json:"inputNeurons"`
	OutputNeurons []*Neuron     `json:"outputNeurons"`
	HiddenNeurons []*Neuron     `json:"hiddenNeurons"`
	Connections   []*Connection `json:"connections"`
}

type Neuron struct {
	ID          int     `json:"id"`
	Name        string  `json:"name"`
	Bias        float64 `json:"bias,omitempty"`
	ActFuncName string  `json:"actFuncName,omitempty"`
	ErrFuncName string  `json:"errFuncName,omitempty"`
}

type Connection struct {
	SrcID  int     `json:"srcId"`
	DstID  int     `json:"dstId"`
	Weight float64 `json:"weight"`
}
