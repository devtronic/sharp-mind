using System;

namespace SharpMind
{
    class SigmoidActivator : IActivator
    {
        public float Activate(float x)
        {
            return (float)(1 / (1 + Math.Exp(-x)));
        }

        public float ActivateDerivative(float x)
        {
            return (float)(Math.Exp(-x) / Math.Pow(1 + Math.Exp(-x), 2));
        }
    }
}
