using System;
using System.Linq;

namespace SharpMind
{
    public class Mind
    {
        public static Mind instance;

        public float netError = 0.0f;

        public static Random rand = new Random(0);

        public IActivator activator;

        public Layer[] layers;


        public Mind(int[] topology, IActivator activator = null)
        {
            if (topology.Count() < 3)
            {
                return;
            }

            if (activator == null)
            {
                activator = new HTanActivator();
            }
            this.activator = activator;

            // Set Number of Layers
            this.layers = new Layer[topology.Count()];

            // Create neurons for input layer
            Layer inputLayer = new Layer();
            inputLayer.neurons = new Neuron[topology[0]];
            for (int iNeuron = 0; iNeuron < topology[0]; iNeuron++)
            {
                inputLayer.neurons[iNeuron] = new Neuron();
            }
            this.layers[0] = inputLayer;
            // End create neurons for input layer

            // Create hidden layers
            for (int iHiddenLayer = 1; iHiddenLayer < topology.Count() - 1; iHiddenLayer++)
            {
                // Create neurons for hidden layer
                Layer hiddenLayer = new Layer();
                hiddenLayer.neurons = new Neuron[topology[iHiddenLayer]];
                for (int hNeuron = 0; hNeuron < topology[iHiddenLayer]; hNeuron++)
                {
                    hiddenLayer.neurons[hNeuron] = new Neuron();
                }
                this.layers[iHiddenLayer] = hiddenLayer;
                // End create neurons for hidden layer
            }
            // End create hidden layers

            // Create neurons for output layer
            Layer outputLayer = new Layer();
            outputLayer.neurons = new Neuron[topology[topology.Count() - 1]];
            for (int oNeuron = 0; oNeuron < topology[topology.Count() - 1]; oNeuron++)
            {
                outputLayer.neurons[oNeuron] = new Neuron();
            }
            this.layers[topology.Count() - 1] = outputLayer;
            // End create neurons for input layer

            this.ConnectNeurons();

            Mind.instance = this;
        }

        public void ConnectNeurons()
        {
            for (int layerIndex = 0; layerIndex < this.layers.Count() - 1; layerIndex++)
            {
                Layer nextLayer = this.layers[layerIndex + 1];

                int synapseCount = nextLayer.neurons.Count();

                for (int neuronIndex = 0; neuronIndex < this.layers[layerIndex].neurons.Count(); neuronIndex++)
                {
                    this.layers[layerIndex].neurons[neuronIndex].synapses = new Synapse[synapseCount];
                    for (int synapseIndex = 0; synapseIndex < synapseCount; synapseIndex++)
                    {
                        Synapse s = new Synapse(Mind.Rand(-2.0f, 2.0f));
                        this.layers[layerIndex].neurons[neuronIndex].synapses[synapseIndex] = s;
                    }
                }
            }
        }

        public bool Propagate(float[] inputValues)
        {
            if (inputValues.Count() != this.layers[0].neurons.Count())
            {
                return false;
            }

            for (int i = 0; i < inputValues.Count(); i++)
            {
                this.layers[0].neurons[i].outputVal = inputValues[i];
            }
            this.FeedForward();
            return true;
        }

        public void FeedForward()
        {
            for (int layerIndex = 1; layerIndex < this.layers.Count(); layerIndex++)
            {
                Layer previousLayer = this.layers[layerIndex - 1];

                this.layers[layerIndex].FeedForward(ref previousLayer);
            }
        }

        public float BackPropagate(float[] expected, float learningRate = 0.2f, float momentum = 0.01f)
        {
            // Prepare Delta Array
            float[][] deltas = new float[this.layers.Count()][];

            for (int layerIndex = 0; layerIndex < deltas.Count(); layerIndex++)
            {
                deltas[layerIndex] = new float[this.layers[layerIndex].neurons.Count()];
            }
            // End Prepare Delta Array

            // Calculate Output Delta
            int lastLayerIndex = this.layers.Count() - 1;
            for (int nO = 0; nO < this.layers[lastLayerIndex].neurons.Count(); nO++)
            {
                float currentOutput = this.layers[lastLayerIndex].neurons[nO].outputVal;
                float error = expected[nO] - currentOutput;
                deltas[lastLayerIndex][nO] = Mind.ActivateDerivative(currentOutput) * error;
            }
            // End Calculate Output Delta

            // Calculate Hidden Delta
            for (int layerIndex = this.layers.Count() - 1; layerIndex > 1; layerIndex--)
            {
                int previousLayerIndex = layerIndex - 1;
                for (int nNext = 0; nNext < this.layers[layerIndex].neurons.Count(); nNext++)
                {
                    for (int nPrevious = 0; nPrevious < this.layers[previousLayerIndex].neurons.Count(); nPrevious++)
                    {
                        float currentOutput = this.layers[previousLayerIndex].neurons[nPrevious].outputVal;
                        float nextDelta = deltas[layerIndex][nNext];
                        float weight = this.layers[previousLayerIndex].neurons[nPrevious].synapses[nNext].weight;
                        float error = nextDelta * weight;

                        deltas[previousLayerIndex][nPrevious] = Mind.ActivateDerivative(currentOutput) * error;
                    }
                }
            }
            // End Calculate Hidden Delta

            // Update Weights
            for (int layerIndex = this.layers.Count() - 1; layerIndex > 0; layerIndex--)
            {
                int previousLayerIndex = layerIndex - 1;
                for (int nNext = 0; nNext < this.layers[layerIndex].neurons.Count(); nNext++)
                {
                    for (int nPrevious = 0; nPrevious < this.layers[previousLayerIndex].neurons.Count(); nPrevious++)
                    {
                        float currentOutput = this.layers[previousLayerIndex].neurons[nPrevious].outputVal;
                        float nextDelta = deltas[layerIndex][nNext];
                        float change = nextDelta * currentOutput;
                        float deltaOld = this.layers[previousLayerIndex].neurons[nPrevious].synapses[nNext].deltaOld;

                        this.layers[previousLayerIndex].neurons[nPrevious].synapses[nNext].weight += (learningRate * change) + (momentum * deltaOld);
                        this.layers[previousLayerIndex].neurons[nPrevious].synapses[nNext].deltaOld = deltaOld;
                    }
                }
            }
            // End Update Weights

            // Calculate Net Error

            float netError = 0.0f;
            for (int k = 0; k < expected.Count(); k++)
            {
                netError += 0.5f * (float)Math.Pow(expected[k] - this.layers[lastLayerIndex].neurons[k].outputVal, 2);
            }
            // End Calculate Net Error
            this.netError = netError;
            return netError;
        }

        public void train(float[][][] patterns, int iterations = 1000, float learningRate = 0.2f, float momentum = 0.01f)
        {
            for (int i = 0; i < iterations; i++)
            {

                foreach (float[][] pattern in patterns)
                {
                    this.Propagate(pattern[0]);
                    this.BackPropagate(pattern[1], learningRate, momentum);
                }
            }
        }

        public float[] getOutput()
        {
            float[] output = new float[this.layers.Last().neurons.Count()];

            for (int n = 0; n < this.layers.Last().neurons.Count(); n++)
            {
                output[n] = this.layers.Last().neurons[n].outputVal;
            }

            return output;
        }

        public static float Rand(float minimum, float maximum)
        {
            return (float)(Mind.rand.NextDouble() * (maximum - minimum) + minimum);
        }

        public static float Activate(float z)
        {
            return Mind.instance.activator.Activate(z);
        }

        public static float ActivateDerivative(float z)
        {
            return Mind.instance.activator.ActivateDerivative(z);
        }
    }

}
