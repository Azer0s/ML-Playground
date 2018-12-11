using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetwork
{
    public class Network
    {
        public readonly List<Layer> Layers = new List<Layer>();
        public readonly Layer InputLayer;
        public readonly Layer OutputLayer;
        private double[] _lastOutput;
        
        public Network(int input, double inputLr, int[] hidden, double[] hiddenLr, int output, double outputLr)
        {
            InputLayer = new Layer(1,input,inputLr);
            OutputLayer = new Layer(hidden.Last(),output,outputLr);
            
            for (var i = 0; i < hidden.Length; i++)
            {
                var inputs = i == 0 ? input : hidden[i - 1];
                Layers.Add(new Layer(inputs,hidden[i],hiddenLr[i]));
            }

            InputLayer.nextAction = doubles => Layers[0].Input(doubles);
            OutputLayer.nextAction = doubles => _lastOutput = _lastOutput = doubles;
            Layers[Layers.Count - 1].nextAction = doubles => OutputLayer.Input(doubles);
            
            for (var k = 0; k < hidden.Length - 1; k++)
            {
                var layer = Layers[k + 1];
                Layers[k].nextAction = doubles => layer.Input(doubles);
            }
        }

        public double[] Predict(double[] input)
        {
            InputLayer.Input(input, true);
            return _lastOutput;
        }

        public static (Network network, double avgError) Train(int input, double inputLr, int[] hidden, double[] hiddenLr, int output, double outputLr, double[][] inputVals, double[][] outputVals)
        {
            double avgError;
            Network network;
            do
            {
                network = new Network(input,inputLr,hidden,hiddenLr,output,outputLr);
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
                        
                        if (errorIndex == errorArray.Length)
                        {
                            errorIndex = 0;
                        }
                        
                        network.InputLayer.Train(error);
                        network.OutputLayer.Train(error);
                        
                        foreach (var layer in network.Layers)
                        {
                            layer.Train(error);
                        }
                        
                        avgError += error;
                    }

                    avgError /= inputVals.Length;
                }
            } while (!(Math.Abs(avgError) < 5E-6) || double.IsNaN(avgError));

            return (network,avgError);
        }
    }
}