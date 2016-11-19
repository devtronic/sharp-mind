using System.Linq;

namespace SharpMind
{
    public class Layer
    {
        public Neuron[] neurons;

        public void FeedForward(ref Layer previousLayer)
        {
            for (int iNeuron = 0; iNeuron < this.neurons.Count(); iNeuron++)
            {
                float sum = 0.0f;

                for (int pNeuron = 0; pNeuron < previousLayer.neurons.Count(); pNeuron++)
                {
                    sum += previousLayer.neurons[pNeuron].outputVal * previousLayer.neurons[pNeuron].synapses[iNeuron].weight;
                }

                sum = Mind.Activate(sum);
                this.neurons[iNeuron].outputVal = sum;
            }
        }

    }
}
