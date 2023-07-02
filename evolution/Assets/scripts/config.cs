
class Config
{
    public static readonly int stepsPerEpoch = 20;
    public static float epochDuration = 1f; // seconds

    public static int fps = (int)(stepsPerEpoch / epochDuration);
    
    public static void SetEpochDuration(float s)
    {
        epochDuration = s;
        CalculateFPS();
    }

    public static void CalculateFPS()
    {
        fps = (int)(stepsPerEpoch / epochDuration);
    }

}
