package com.example.suvam.myapplication;

import java.io.File;
import java.io.FileOutputStream;

import android.content.Context;
import android.hardware.Camera;
import android.hardware.Camera.PictureCallback;
import android.os.Environment;
import android.widget.Toast;

public class PhotoHandler implements PictureCallback {

    private final Context context;
    private int imid;
    private String baseName;
    private int lagFlag[];

    public PhotoHandler(Context context, int id, String bn, int [] lf) {
        this.context = context;
        this.imid = id;
        this.baseName = bn;
        this.lagFlag = lf;
    }

    @Override
    public void onPictureTaken(byte[] data, Camera camera) {

        File pictureFileDir = new File (Environment.getExternalStorageDirectory().getAbsolutePath() + "/SensorFusion3");
        if (!pictureFileDir.exists() && !pictureFileDir.mkdirs())
        {
            Toast.makeText(context, "Can't create directory to save image.",
                    Toast.LENGTH_LONG).show();
            return;

        }

        String photoFile = "SensorFusion3"  +  imid + ".jpg";
        String filename = pictureFileDir +"/" + photoFile;

        File pictureFile = new File(filename);

        try {
            FileOutputStream fos = new FileOutputStream(pictureFile);
            fos.write(data);
            fos.close();
            lagFlag[0] = 0;
            Toast.makeText(context, "Image ID " + imid + " saved",
                Toast.LENGTH_SHORT).show();
        } catch (Exception error) {
            Toast.makeText(context, "Unable to save the clicked image.",
                    Toast.LENGTH_SHORT).show();
        }
        camera.startPreview();
    }
}
