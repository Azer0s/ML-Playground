using System;
using System.Linq;

namespace NeuralNetwork
{
    public class Layer
    {
        private readonly Perceptron[] _perceptrons;
        public Action<double[]> NextAction;

        public Layer(int nrOfInputs, int nrOfPerceptrons, double lr, Action<double[]> nextAction = null)
        {
            _perceptrons = new Perceptron[nrOfPerceptrons];

            for (var i = 0; i < _perceptrons.Length; i++) _perceptrons[i] = new Perceptron(nrOfInputs, lr);

            NextAction = nextAction;
        }

        public void Input(double[] input, bool treatAsSingle = false)
        {
            NextAction(!treatAsSingle
                ? _perceptrons.Select(perceptron => perceptron.Predict(input)).ToArray()
                : input.Select((t, i) => _perceptrons[i].Predict(new[] {t})).ToArray());
        }

        public void Train(double error)
        {
            foreach (var perceptron in _perceptrons) perceptron.Train(error);
        }
    }
}