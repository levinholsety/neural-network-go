package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"time"

	"github.com/levinholsety/common-go/num"
	"github.com/levinholsety/neural-network/nn"
)

func main() {
	switch os.Args[1] {
	case "1":
		data := train1()
		ioutil.WriteFile(`D:\nn_xor.json`, data, 0644)
		run1(data)
	case "11":
		data, _ := ioutil.ReadFile(`D:\nn_xor.json`)
		run1(data)
	case "2":
		ioutil.WriteFile(`D:\nn_xor.json`, train2(), 0644)
	case "22":
		data, _ := ioutil.ReadFile(`D:\nn_xor.json`)
		run2(data)
	}
}

func train1() []byte {
	actFuncName := nn.AFSigmoid
	inputLayer := nn.NewInputLayer(2)
	hiddenLayer := nn.NewHiddenLayer(2, 0, 0.35, actFuncName)
	outputLayer := nn.NewOutputLayer(2, 0.6, actFuncName, nn.EFMSE)

	net := nn.NewNeuralNetwork()

	net.ConnectLayers(inputLayer, hiddenLayer, num.Matrix{
		{0.15, 0.25},
		{0.2, 0.3},
	})
	net.ConnectLayers(hiddenLayer, outputLayer, num.Matrix{
		{0.4, 0.5},
		{0.45, 0.55},
	})

	net.SetPatterns(num.Matrix{
		{0.05, 0.1, 0.01, 0.99},
	})

	net.SetLearning(true)
	net.SetLearningRate(0.5)
	net.Go()
	fmt.Printf("TE: %.10f\n", net.TotalError)
	for i := 0; i < 9999; i++ {
		net.Go()
		if i%1000 == 0 {
			fmt.Printf("TE: %.10f\n", net.TotalError)
		}
	}
	fmt.Printf("TE: %.10f\n", net.TotalError)

	net.SetLearning(false)

	net.SetPatterns(num.Matrix{
		{0.05, 0.1},
	})
	net.Go()
	net.PrintOutputValues()

	buf := &bytes.Buffer{}
	net.Save(buf)
	return buf.Bytes()
}

func run1(data []byte) {
	net := nn.NewNeuralNetwork()
	err := net.Load(bytes.NewReader(data))
	if err != nil {
		fmt.Println(err)
	}
	net.SetPatterns(num.Matrix{
		{0.05, 0.1},
		{0.06, 0.09},
		{0.04, 0.11},
		{0.04, 0.09},
		{0.06, 0.11},
	})
	net.Go()
	net.PrintOutputValues()
}

var r = rand.New(rand.NewSource(time.Now().UnixNano()))

func weight() float64 {
	return r.NormFloat64()
}

func train2() []byte {
	actFuncName := nn.AFSigmoid
	ic, hc, oc := 2, 3, 1
	inputLayer := nn.NewInputLayer(ic)
	hiddenLayer := nn.NewHiddenLayer(hc, 0, 0, actFuncName)
	outputLayer := nn.NewOutputLayer(oc, 0, actFuncName, nn.EFMSE)

	net := nn.NewNeuralNetwork()

	net.ConnectLayers(inputLayer, hiddenLayer, num.NewMatrix(num.NewMatrixSize(ic, hc)).InitRandN())
	net.ConnectLayers(hiddenLayer, outputLayer, num.NewMatrix(num.NewMatrixSize(hc, oc)).InitRandN())

	net.SetLearning(true)
	net.SetLearningRate(0.8)
	net.SetMomentum(0.3)

	net.SetPatterns(num.Matrix{
		{0, 0, 0},
		{0, 1, 1},
		{1, 0, 1},
		{1, 1, 0},
	})
	i := 0
	for {
		i++
		net.Go()
		if net.TotalError < 0.01 {
			break
		}
		if i%10000 == 0 {
			fmt.Printf("TE: %.10f\n", net.TotalError)
		}
	}
	fmt.Printf("TE: %.10f\n", net.TotalError)
	fmt.Println(i)

	buf := &bytes.Buffer{}
	net.Save(buf)
	return buf.Bytes()
}

func run2(data []byte) {
	net := nn.NewNeuralNetwork()
	net.Load(bytes.NewReader(data))
	net.SetPatterns(num.Matrix{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
		{0.1, 0.1},
		{0.1, 0.9},
		{0.9, 0.1},
		{0.9, 0.9},
	})
	net.Go()
	net.PrintOutputValues()
}
