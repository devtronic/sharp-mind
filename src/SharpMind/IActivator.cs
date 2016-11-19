namespace SharpMind
{
    public interface IActivator
    {
        float Activate(float x);

        float ActivateDerivative(float x);
    }
}
