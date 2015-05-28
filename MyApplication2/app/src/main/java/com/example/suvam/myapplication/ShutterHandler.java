package com.example.suvam.myapplication;


import android.content.Context;
import android.hardware.Camera;
import android.hardware.Camera.ShutterCallback;
import android.widget.Toast;

public class ShutterHandler implements ShutterCallback {

    private final Context context;
    private int imid;
    Camera c;

    public ShutterHandler(Context context, int id, Camera cam) {
        this.context = context;
        this.imid = id;
        this.c = cam;
    }

    @Override
    public void onShutter()
    {
        Toast.makeText(context, "Image ID " + imid + " clicked", Toast.LENGTH_SHORT).show();
        c.startPreview();
    }
}