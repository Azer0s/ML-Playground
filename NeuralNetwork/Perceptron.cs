using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Perceptron
    {
        private readonly double[] _weights;
        private double[] _lastInputs;
        private readonly double _lr;
        private double _bias;
        
        public Perceptron(int nrOfInputs, double lr)
        {
            _weights = new double[nrOfInputs];
            _bias = new Random().NextDouble();

            for (var i = 0; i < nrOfInputs; i++)
            {
                _weights[i] = new Random().NextDouble();
            }
            _lr = lr;
        }

        public double Predict(IEnumerable<double> input)
        {
            _lastInputs = input.ToArray();
            return input.Select((t, i) => _weights[i] * t).Sum() + _bias;
        }

        public void Train(double error)
        {
            for (var i = 0; i < _weights.Length; i++)
            {
                var gradient = _lastInputs[i] * error * _lr;
                _weights[i] += _weights[i] * gradient;
            }

            _bias += error * _lr;
        }
    }
}