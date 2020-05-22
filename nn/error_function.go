package nn

import "github.com/levinholsety/common-go/num"

func NewErrorFunction(name string) ErrorFunction {
	return efMap[name]
}

type ErrorFunction interface {
	Func(target, output num.Tensor) num.Tensor
	PDFunc(target, output num.Tensor) num.Tensor
}

const (
	EFMSE          = "MSE"
	EFCrossEntropy = "CrossEntropy"
)

var efMap = map[string]ErrorFunction{
	EFMSE:          new(MSE),
	EFCrossEntropy: new(CrossEntropy),
}

type MSE struct{}

func (p *MSE) Func(target, output num.Tensor) num.Tensor {
	return target.Sub(output).Square()
}

func (p *MSE) PDFunc(target, output num.Tensor) num.Tensor {
	return target.Sub(output).Negative()
}

type CrossEntropy struct{}

func (p *CrossEntropy) Func(target, output num.Tensor) num.Tensor {
	return target.Mul(output.Log()).Negative()
}

func (p *CrossEntropy) PDFunc(target, output num.Tensor) num.Tensor {
	return target.Div(output).Negative().Add(num.Scalar(1).Sub(target).Div(num.Scalar(1).Sub(output)))
}
