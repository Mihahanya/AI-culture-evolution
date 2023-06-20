using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class camera : MonoBehaviour
{
    public float speed = 30f;
    public float nonactiveArea = 0.2f;

    public float scrollSpeed = 3f;

    public float minFov = 3f, maxFov = 40f;

    public bool pause = false;

    private float camera_size = 5;
    private float lastFrameTime;


    void Start()
    {
        camera_size = Camera.main.orthographicSize;

        lastFrameTime = Time.realtimeSinceStartup;
    }

    void LateUpdate()
    {
        float myDeltaTime = Time.realtimeSinceStartup - lastFrameTime;
        lastFrameTime = Time.realtimeSinceStartup;

        Vector2 screen_sizes = new Vector2(Screen.width, Screen.height);
        Vector2 mouse = new Vector2(Input.mousePosition.x, Input.mousePosition.y);
        Vector2 mouse_rel = mouse - screen_sizes / 2f;
        mouse_rel /= screen_sizes / 2f;

        // Zoom

        camera_size += -Input.GetAxis("Mouse ScrollWheel") * scrollSpeed * camera_size;
        camera_size = Mathf.Clamp(camera_size, minFov, maxFov);
        Camera.main.orthographicSize = camera_size;

        // Pause

        if (Input.GetKeyDown(KeyCode.P) && !pause)
        {
            pause = true;
            Time.timeScale = 0;
        }
        else if (Input.GetKeyDown(KeyCode.P) && pause)
        {
            pause = false;
            Time.timeScale = 1;
        }

        // Move

        Display dispData = GetComponent<Display>();

        if (dispData.isFollowObject && dispData.followingObject != null)
        {
            var followingObject = dispData.followingObject;

            Vector3 move_to = new Vector3(followingObject.transform.position.x,
                                          followingObject.transform.position.y,
                                          transform.position.z);

            transform.position = Vector3.Lerp(transform.position, move_to, 7f * Time.deltaTime);
        }
        else if (mouse_rel.magnitude > nonactiveArea && Application.isFocused)
        {
            transform.Translate(mouse_rel / mouse_rel.magnitude
                                * (mouse_rel.magnitude-nonactiveArea) 
                                * speed * myDeltaTime);
        }


        if (Input.GetKeyDown(KeyCode.T)) transform.position = new Vector3(0, 0, transform.position.z);

    }
}
