namespace SharpMind
{
    public class Synapse
    {
        public float weight = 0.0f;

        public float deltaOld = 0.0f;

        public Synapse(float weight)
        {
            this.weight = weight;
        }
    }
}
