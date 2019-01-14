using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Network
    {
        private readonly Layer _inputLayer;
        private readonly List<Layer> _layers = new List<Layer>();
        private readonly Layer _outputLayer;
        private double[] _lastOutput;

        private Network(int input, double inputLr, IReadOnlyList<int> hidden, IReadOnlyList<double> hiddenLr, int output, double outputLr)
        {
            _inputLayer = new Layer(1, input, inputLr);
            _outputLayer = new Layer(hidden.Last(), output, outputLr);

            for (var i = 0; i < hidden.Count; i++)
            {
                var inputs = i == 0 ? input : hidden[i - 1];
                _layers.Add(new Layer(inputs, hidden[i], hiddenLr[i]));
            }

            _inputLayer.NextAction = doubles => _layers[0].Input(doubles);
            _outputLayer.NextAction = doubles => _lastOutput = _lastOutput = doubles;
            _layers[_layers.Count - 1].NextAction = doubles => _outputLayer.Input(doubles);

            for (var k = 0; k < hidden.Count - 1; k++)
            {
                var layer = _layers[k + 1];
                _layers[k].NextAction = doubles => layer.Input(doubles);
            }
        }

        public double[] Predict(double[] input)
        {
            _inputLayer.Input(input, true);
            return _lastOutput;
        }

        public static (Network network, double avgError) Train(int input, double inputLr, int[] hidden, double[] hiddenLr, int output, double outputLr, double[][] inputVals, double[][] outputVals)
        {
            double avgError;
            Network network;
            do
            {
                network = new Network(input, inputLr, hidden, hiddenLr, output, outputLr);
                avgError = 0.0;
                var errorIndex = 0;

                for (var j = 0; j < 100000; j++)
                {
                    for (var i = 0; i < inputVals.Length; i++)
                    {
                        var prediction = network.Predict(inputVals[i]);
                        var errorArray = outputVals[i].Select((d, index) => d - prediction[index]).ToArray();
                        var error = errorArray[errorIndex];

                        errorIndex++;

                        if (errorIndex == errorArray.Length) errorIndex = 0;

                        network._inputLayer.Train(error);
                        network._outputLayer.Train(error);

                        foreach (var layer in network._layers) layer.Train(error);

                        avgError += error;
                    }

                    avgError /= inputVals.Length;
                }
            } while (!(Math.Abs(avgError) < 5E-6) || double.IsNaN(avgError));

            return (network, avgError);
        }
    }
}