using System;
using System.Linq;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            var inputs = new[]
            {
                new double[] {1, 1, 1, 1},
                new double[] {2, 3, 4, 5},
                new double[] {5, 8, 9, 9},
                new double[] {13, 21, 22, 7},
                new double[] {4, 4, 5, 6}, 
                new double[] {5, 5, 6, 7},
                new double[] {4, 8, 9, 3}
            };
            var outputs = new []{5.1, 14.5, 35.9, 90.7, 21.6, 26.7, 33.3};

            var avgError = double.MaxValue;
            double[] prediction = null;
            Layer outputLayer;
            Layer hiddenLayer2;
            Layer hiddenLayer1;
            Layer inputLayer;

            do
            {
                avgError = 0.0;
                outputLayer = new Layer(3, 1, 0.0001, doubles => prediction = doubles);
                hiddenLayer2 = new Layer(6, 3, 0.001, doubles => outputLayer.Input(doubles));
                hiddenLayer1 = new Layer(4, 6, 0.0001, doubles => hiddenLayer2.Input(doubles));
                inputLayer = new Layer(1, 4, 0.0001, doubles => hiddenLayer1.Input(doubles));

                for (var j = 0; j < 100000; j++)
                {
                    for (var i = 0; i < inputs.Length; i++)
                    {
                        inputLayer.Input(inputs[i], true);
                        var error = outputs[i] - prediction.First();
                        outputLayer.Train(error);
                        hiddenLayer2.Train(error);
                        hiddenLayer1.Train(error);
                        inputLayer.Train(error);
                        avgError = error;
                    }

                    avgError /= inputs.Length;
                }
            } while (!(Math.Abs(avgError) < 5E-6) || double.IsNaN(avgError));
            
            inputLayer.Input(new double[]{10, 10, 5, 4},true);
            Console.WriteLine(prediction.First());
        }
    }
}