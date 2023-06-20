using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class spawner : MonoBehaviour
{
    public GameObject obj;
    public float countNewInSec = 1;
    public float spreadX = 10, spreadY = 10;

    public int prespawnedCount = 10;
    public int maxCount = 100;
    public string tagToCount;

    private float timer;

    void Start()
    {
        timer = 0.5f;

        for (int i = 0; i < prespawnedCount; i++)
        {
            spawn_another();
        }
    }

    void LateUpdate()
    {
        timer -= Time.deltaTime;
        if (timer <= 0)
        {
            int alreadyCount = GameObject.FindGameObjectsWithTag(tagToCount).Length;

            for (int i=0; i < Mathf.Min(countNewInSec/2, maxCount-alreadyCount); i++) spawn_another();
            timer = 0.5f;
        }
    }

    private void spawn_another()
    {
        obj.transform.position = new Vector3(Random.Range(-spreadX, spreadX) + transform.position.x,
                                             Random.Range(-spreadY, spreadY) + transform.position.y,
                                             obj.transform.position.z);
        Instantiate(obj);
    }
}
