using System;

namespace SharpMind
{
    class HTanActivator : IActivator
    {
        public float Activate(float x)
        {
            float y = (float)Math.Exp(2 * x);
            return (float)(y - 1) / (y + 1);
        }

        public float ActivateDerivative(float x)
        {
            return (float)(1 - Math.Pow((Math.Exp(2 * x) - 1) / (Math.Exp(2 * x) + 1), 2));
        }
    }
}
