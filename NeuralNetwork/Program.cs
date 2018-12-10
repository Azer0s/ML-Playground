using System;
using System.Linq;

namespace NeuralNetwork
{
    class Program
    {
        static void Main()
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

            var (network, avgError) = Network.Train(4, 0.0001, new[] {6, 3}, new[] {0.0001, 0.001}, 1, 0.0001, inputs,outputs);
            Console.WriteLine($"Error: {avgError}");
            Console.WriteLine(network.Predict(new double[]{10, 10, 5, 4}).First());
        }
    }
}