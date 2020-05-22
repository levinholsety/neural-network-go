package nn

import (
	"github.com/levinholsety/common-go/num"
)

const (
	AFSigmoid = "sigmoid"
	AFTanh    = "tanh"
)

var afMap = map[string]ActivationFunction{
	AFSigmoid: new(Sigmoid),
	AFTanh:    new(Tanh),
}

func NewActivationFunction(name string) ActivationFunction {
	return afMap[name]
}

type ActivationFunction interface {
	Func(x num.Tensor) num.Tensor
	PDFunc(x num.Tensor) num.Tensor
}

type Sigmoid struct{}

func (p *Sigmoid) Func(x num.Tensor) num.Tensor {
	return num.Scalar(1).Div(num.Scalar(1).Add(x.Negative().Exp()))
}

func (p *Sigmoid) PDFunc(x num.Tensor) num.Tensor {
	return x.Mul(num.Scalar(1).Sub(x))
}

type Tanh struct{}

func (p *Tanh) Func(x num.Tensor) num.Tensor {
	return x.Exp().Sub(x.Negative().Exp()).Div(x.Exp().Add(x.Negative().Exp()))
}

func (p *Tanh) PDFunc(x num.Tensor) num.Tensor {
	return num.Scalar(1).Sub(x.Square())
}
