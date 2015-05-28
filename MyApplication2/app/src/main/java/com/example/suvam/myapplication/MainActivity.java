package com.example.suvam.myapplication;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.*;

import android.hardware.Camera;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.CamcorderProfile;
import android.media.MediaMetadataRetriever;
import android.media.MediaRecorder;
import android.media.MediaRecorder.OnErrorListener;
import android.media.MediaRecorder.OnInfoListener;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.app.Activity;
import android.content.Intent;
import android.util.Log;
import android.view.Menu;
import android.view.MotionEvent;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.MediaController;
import android.widget.RadioButton;
import android.widget.TextView;
import android.widget.Toast;
import android.widget.VideoView;
import android.location.Criteria;
import android.location.Location;
import android.location.LocationListener;
import android.location.LocationManager;
import android.content.Context;

import Jama.*;
import java.util.Date;

public class MainActivity extends Activity implements SensorEventListener, SurfaceHolder.Callback, MediaRecorder.OnInfoListener, MediaRecorder.OnErrorListener
{  private SensorManager mSensorManager = null;

    // angular speeds from gyro
    private float[] gyro = new float[3];

    // rotation matrix from gyro data
    private float[] gyroMatrix = new float[9];

    // orientation angles from gyro matrix
    private float[] gyroOrientation = new float[3];


    // magnetic field vector
    private float[] magnet = new float[3];

    // gravity vector
    private float[] grav = new float[3];

    private int count1;

    // accelerometer vector
    private float[] accel = new float[3];

    // orientation angles from grav and magnet
    private float[] accMagOrientation = new float[3];

    // final orientation angles from sensor fusion
    private float[] fusedOrientation = new float[3];

    // accelerometer and magnetometer based rotation matrix
    private float[] rotationMatrix = new float[9];

    public static final float EPSILON = 0.000000001f;
    private static final float NS2S = 1.0f / 1000000000.0f;
    private float timestamp;
    public static final float FILTER_COEFFICIENT = 0.98f;

    TextView mess = null;
    public double[][] temp;


    float magVals[];
    float gravVals[];
    float accVals[];
    float rotMat[];
    float orientation[];
    float univAcc[];

    //Timings
    long startStamp = 0;
    long nowStamp = 0;
    boolean firstFixDone;
    boolean appStarted;
    int imageId;
    int isLagData[];

    BufferedWriter file=null;
    String fileS=null;
    String fileS2=null;
    String current=null;

    boolean loggingActive;
    boolean toWrite2;
    boolean initState;
    long curtime;

    private int counter;

    File fileDir = null;
    Date date = null;
    SimpleDateFormat ft = null;
    SimpleDateFormat ftTime = null;
    Button startb = null;
    Button aboutb = null;
    Button quitb = null;
    Button click = null;
    Button clearfiles=null;
    EditText name = null;

    String fileSnew;

//    Camera mCamera = null;
//    SurfaceHolder mHolder = null;
//    MediaRecorder mediaRecorder;
    VideoView mVideoView = null;
    boolean recording;
    float rad2deg = 180.0f/(float)Math.PI;

    Vector ax;
    Vector ay;
    Vector az;

    Vector gx;
    Vector gy;
    Vector gz;

    Vector iid;

    Vector<float[]> rotmat1;
    Vector<float[]> rotmat2;

    Vector timev;
    Vector timesum;

    public double[][] Yx;
    public double[][] Yy;
    public double[][] Yz;
    public double[][] X ;
    Matrix tempx, XT, MatrixYx, MatrixYy, MatrixYz;

    private LocationManager locationManager;
    private String provider;
    Location location;

    double lat;
    double lng;
    double alti;
    float acc;

    float alpha;
    boolean whatever;
    boolean firstacc;
    boolean firstgrav;
    boolean firstmag;

    private final String VIDEO_PATH_NAME = Environment.getExternalStorageDirectory().getAbsolutePath() + "/test1.mp4";

    private MediaRecorder mMediaRecorder;
    private Camera mCamera;
    private SurfaceView mSurfaceView;
    private SurfaceHolder mHolder;
    private View mToggleButton;
    private boolean mInitSuccesful;

    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        recording = false;



        setContentView(R.layout.activity_main);

        Log.d("test", "lets see if this comes in\n");
        magVals = new float[3];
        rotMat = new float[9];
        gravVals = new float[3];
        accVals = new float[3];
        orientation = new float[3];
        univAcc = new float[3];

        accel[0]=accel[1]=accel[2]=grav[0]=grav[1]=grav[2]=magnet[0]=magnet[1]=magnet[2]=0.0f;

        alpha=0.05f;
        firstacc=true;
        firstgrav=true;
        firstmag=true;

        isLagData = new int[1];
        isLagData[0] = 0;

        ax=new Vector();
        ay=new Vector();
        az=new Vector();

        count1=0;

        gx=new Vector();
        gy=new Vector();
        gz=new Vector();

        rotmat1=new Vector<float[]>();
        rotmat2=new Vector<float[]>();

        iid = new Vector();
        timev=new Vector();
        timesum=new Vector();

        firstFixDone = false;
        appStarted = false;

        initState=true;
        toWrite2=false;

        counter=1;

        mSensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        mSensorManager.registerListener(this,
                mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER),
                SensorManager.SENSOR_DELAY_FASTEST);
        mSensorManager.registerListener(this,
                mSensorManager.getDefaultSensor(Sensor.TYPE_GRAVITY),
                SensorManager.SENSOR_DELAY_FASTEST);

        mSensorManager.registerListener(this,
                mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE),
                SensorManager.SENSOR_DELAY_FASTEST);

        mSensorManager.registerListener(this,
                mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD),
                SensorManager.SENSOR_DELAY_FASTEST);




        mess = (TextView) findViewById(R.id.mess);

        fileDir = new File (Environment.getExternalStorageDirectory().getAbsolutePath() + "/SensorFusion3");
        fileDir.mkdirs();

        Log.d ("test", "Sensors initialised");
        //Logging
        ft = new SimpleDateFormat("hh_mm_ss'@'dd_MM_yyyy", Locale.ENGLISH);
        ftTime = new SimpleDateFormat("hh:mm:ss", Locale.ENGLISH);

        startb = (Button) findViewById(R.id.startButtn);
        startb.setText(getString(R.string.start_button));

        quitb = (Button) findViewById(R.id.quitButtn);
        quitb.setText("Quit");

        click = (Button) findViewById(R.id.clickButtn);
        click.setText("Image Click");

        clearfiles =(Button) findViewById(R.id.cbutton);
        clearfiles.setText("Clear");

        mVideoView = (VideoView)this.findViewById(R.id.videoView);
        initCamera();


        aboutb = (Button) findViewById(R.id.aboutButtn);

        startb.setOnClickListener(new View.OnClickListener()
        {
            public void onClick(View v)
            {
                if(!appStarted)
                {
                    try
                    {
                        Log.d("test","In the try for start button");
                        counter=0;
                        click.setClickable(true);

                        CheckBox radiolistener = (CheckBox) findViewById(R.id.checkBox1);
                        whatever= radiolistener.isChecked();


                        date = new Date();
                        appStarted = true;
                        isLagData[0] = 0;
                        startb.setText(getString(R.string.stop_button));
                        current = DateFormat.getDateTimeInstance().format(new Date());
                        curtime= new Date().getTime();
                        fileS = fileDir + "/"+  curtime +  "SensorFusion3.csv";

                        file = new BufferedWriter(new FileWriter(fileS));

                        file.write("Time,Rot1,Rot2,Rot3,Rot4,Rot5,Rot6,Rot7,Rot8,Rot9,FRot1,FRot2,FRot3,FRot4,FRot5,FRot6,FRot7,FRot8,FRot9,AccX,AccY,AccZ,GravX,GravY,GravZ,GyroX,GyroY,GyroZ,MagX,MagY,MagZ,Imid\n");

                        file.flush();
//                        file.close();

                        Log.d("test","File initialised");

                        loggingActive = true;
                        mess.setText("Started logging to " + fileS);
                        imageId = 0;
                        toWrite2=true;
                        Log.d("test", "to write value changed");

                        List<Sensor> senslist = mSensorManager.getSensorList(Sensor.TYPE_ALL);
                        fileSnew = fileDir + "/"+  "SensorFusion3.txt";

                        BufferedWriter filenew = new BufferedWriter(new FileWriter(fileSnew));
                        filenew.write("Name \t Vendor \t Resolution \t MinDelay \t Version \n ");
                        for (int i=0; i< senslist.size() ; i++)
                        {
                            filenew.write(senslist.get(i).getName()+ "\t" + senslist.get(i).getVendor() + "\t" + senslist.get(i).getResolution()+ "\t" + senslist.get(i).getMinDelay()+ "\t" + senslist.get(i).getVersion() + "\n");
                            filenew.flush();
                        }
                        filenew.close();


                    }

                    catch (IOException e)
                    {
                        Log.d("test","In catch of start button, file could not be created");
                        mess.setText("Error processing file.");
                        e.printStackTrace();
                    }

                    Log.d("test","started video");
//                    try {
//                        mediaRecorder.prepare();
////                        Thread.sleep(1000);
//                    } catch (IOException e) {
//                        e.printStackTrace();
//                    }
//                    if(mediaRecorder)
//                    mMediaRecorder.start();
                    Log.d("test", "finished video");
                }
                else
                {
                    Log.d("test", "Starting stop actions");
//                    mMediaRecorder.stop();
//                    mMediaRecorder.release();
//                    finish();
                    toWrite2=false;
//                    try {
//                        Thread.sleep(1);
//                    } catch (InterruptedException e) {
//                        e.printStackTrace();
//                    }
                    startb.setText(getString(R.string.start_button));
                    appStarted = false;
                    loggingActive = false;
                    firstFixDone = false;
                    click.setClickable(false);
                    Log.d("test", "booleans modified");
                    mess.setText("Stopped logging to " + fileS + "\nProccessing data now");
                    Log.d("test", "starting file close");
                    try
                    {
                        file.close();
                    }
                    catch (IOException e)
                    {
                        e.printStackTrace();
                    }
                    Log.d("test", "done file closed");
                    ProcessData();
                    mess.setText("Done!");
                    final Intent intent = new Intent(Intent.ACTION_SEND_MULTIPLE);
                    intent.setType("text/plain");
                    intent.putExtra(android.content.Intent.EXTRA_EMAIL,
                            new String[]{"kartikeyagupta@hotmail.com"});
                    intent.putExtra(Intent.EXTRA_SUBJECT, "Data");
                    intent.putExtra(Intent.EXTRA_TEXT, "PFA");

                    ArrayList<Uri> uris = new ArrayList<Uri>();
                    File fileIn = new File(fileS);
                    Uri u = Uri.fromFile(fileIn);
                    uris.add(u);
                    File fileIn2 = new File(fileS2);
                    Uri u2 = Uri.fromFile(fileIn2);
                    uris.add(u2);
                    File fileIn3 = new File(fileSnew);
                    Uri u3 = Uri.fromFile(fileIn3);
                    uris.add(u3);

                    intent.putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris);

                    startActivity(Intent.createChooser(intent, "Choose an Email client :"));
                }
            }
        });

        aboutb.setOnClickListener(new View.OnClickListener()
        {
            public void onClick(View v)
            {
                Intent i = new Intent(MainActivity.this, About.class);
                startActivity(i);
            }
        });

        clearfiles.setOnClickListener(new View.OnClickListener()
        {
            public void onClick(View v)
            {
                File dir = new File(Environment.getExternalStorageDirectory()+"/SensorFusion3");
                if (dir.isDirectory()) {
                    String[] children = dir.list();
                    for (int i = 0; i < children.length; i++) {
                        new File(dir, children[i]).delete();
                    }
                }
                Toast.makeText(getBaseContext(),
                        "All Files Deleted!",
                        Toast.LENGTH_SHORT).show();
            }

        });

        quitb.setOnClickListener(new View.OnClickListener()
        {
            public void onClick(View v)
            {
                Toast.makeText(getBaseContext(),
                        "Thank You!",
                        Toast.LENGTH_SHORT).show();
                finish();
            }
        });

        click.setOnClickListener(new View.OnClickListener()
        {
            private FocusCallback autoF = new FocusCallback(mCamera);
            public void onClick(View v)
            {
                isLagData[0] = 1;
                mess.setText("Image Id clicked: " + String.valueOf(imageId));
                autoF.preparePicture(new PhotoHandler(getApplicationContext(), imageId+1, current,isLagData));
                mCamera.autoFocus(autoF);
                imageId++;
            }
        });
        click.setClickable(false);
    }



    public void onSensorChanged(SensorEvent event) {
//        Log.d ("sense","Sensor Event");
        if (toWrite2) {
            count1++;
            switch (event.sensor.getType()) {
                case Sensor.TYPE_ACCELEROMETER:
//                mess.setText("acc");
//                    Log.d ("sense",count1+ " accel event");
                    if (whatever)
                        System.arraycopy(event.values, 0, accel, 0, 3);
                    else
                    {
//                        if(firstacc)
//                        {
                            System.arraycopy(event.values, 0, accel, 0, 3);
//                            firstacc=false;
//                        }
//                        else
//                        {
//                            accel[0] = (1 - alpha) * accel[0] + alpha * event.values[0];
//                            accel[1] = (1 - alpha) * accel[1] + alpha * event.values[1];
//                            accel[2] = (1 - alpha) * accel[2] + alpha * event.values[2];
//                        }
                    }
                    break;
                case Sensor.TYPE_GRAVITY:
                mess.setText("grav");
                    // copy new accelerometer data into grav array and calculate orientation
                    if (whatever)
                        System.arraycopy(event.values, 0, grav, 0, 3);
                    else
                    {
//                        if(firstgrav)
//                        {
                            System.arraycopy(event.values, 0, grav, 0, 3);
//                            firstgrav=false;
//                        }
//                        else
//                        {
//                            grav[0] = (1 - alpha) * grav[0] + alpha * event.values[0];
//                            grav[1] = (1 - alpha) * grav[1] + alpha * event.values[1];
//                            grav[2] = (1 - alpha) * grav[2] + alpha * event.values[2];
//                        }

                    }
                calculateAccMagOrientation();
//                    Log.d ("sense",count1+ " gravity event");

                    if (toWrite2) {
                        mess.setText("if");
                        mess.setText("sensor event writing case" + count1);
                        if (counter == 0) {
                            startStamp = event.timestamp;
                            counter = 1;
                        }
                        try {

//                            onLocationChanged(location);
                            file.write(((event.timestamp - startStamp) / 1e+9) + "," +
                                    rotationMatrix[0] + "," + rotationMatrix[1] + "," + rotationMatrix[2] + "," +
                                    rotationMatrix[3] + "," + rotationMatrix[4] + "," + rotationMatrix[5] + "," +
                                    rotationMatrix[6] + "," + rotationMatrix[7] + "," + rotationMatrix[8] + "," +
                                    gyroMatrix[0] + "," + gyroMatrix[1] + "," + gyroMatrix[2] + "," +
                                    gyroMatrix[3] + "," + gyroMatrix[4] + "," + gyroMatrix[5] + "," +
                                    gyroMatrix[6] + "," + gyroMatrix[7] + "," + gyroMatrix[8] + "," +
                                    (accel[0]) + "," + (accel[1]) + "," + (accel[2]) + "," +
                                    grav[0] + "," + grav[1] + "," + grav[2] + "," +
                                    gyro[0] + "," + gyro[1] + "," + gyro[2] + "," +
                                    magnet[0] + "," + magnet[1] + "," + magnet[2] + "," +
                                    imageId + "," +
                                    acc + "," + lat + "," + lng + "," + alti + "\n");
                            file.flush();

                            ax.addElement(new Float(accel[0]));
                            ay.addElement(new Float(accel[1]));
                            az.addElement(new Float(accel[2]));

                            gx.addElement(new Float(grav[0]));
                            gy.addElement(new Float(grav[1]));
                            gz.addElement(new Float(grav[2]));

                            timev.addElement(new Long(event.timestamp - startStamp));
                            timesum.addElement(new Long(event.timestamp - startStamp));
                            iid.addElement(new Integer(imageId));

                            rotmat1.addElement(rotationMatrix);
                            rotmat2.addElement(gyroMatrix);


                        } catch (IOException e) {
                            mess.setText("Unable to write data to file");
                            e.printStackTrace();
                        }
                    } else {
                    mess.setText("Else"+count1);
                    }
                    break;

                case Sensor.TYPE_GYROSCOPE:
//                    Log.d ("sense",count1+ " gyro event");
                    // process gyro data
//                mess.setText("gyro");
                    gyroFunction(event);
                    break;

                case Sensor.TYPE_MAGNETIC_FIELD:
//                    Log.d ("sense",count1+ " magnetic event");
                    // copy new magnetometer data into magnet array
//                mess.setText("mag");
                    if (whatever)
                        System.arraycopy(event.values, 0, magnet, 0, 3);
                    else
                    {
//                        if(firstmag)
//                        {
                            System.arraycopy(event.values, 0, magnet, 0, 3);
//                            firstmag=false;
//                        }
//                        else
//                        {
//                            magnet[0] = (1 - alpha) * magnet[0] + alpha * event.values[0];
//                            magnet[1] = (1 - alpha) * magnet[1] + alpha * event.values[1];
//                            magnet[2] = (1 - alpha) * magnet[2] + alpha * event.values[2];
//                        }
                    }
                    break;
            }
        }
    }



    public void onAccuracyChanged(Sensor sensor, int accuracy)
    {
    }


    @Override
    public void surfaceCreated(SurfaceHolder holder)
    {
        try
        {
            mCamera.setPreviewDisplay(mHolder);
            mCamera.startPreview();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
//        try {
//            if (!mInitSuccesful)
//                initRecorder(mHolder.getSurface());
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        prepareMediaRecorder();
    }


    private void initRecorder(Surface surface) throws IOException {
        Log.d("test","starting initrecorder");
        // It is very important to unlock the camera before doing setCamera
        // or it will results in a black preview
        if (mCamera == null) {
            Log.d("test","in null case");
            mCamera = Camera.open();
            mCamera.unlock();
        }
        Log.d("test","camera obtained");


        if (mMediaRecorder == null) mMediaRecorder = new MediaRecorder();
//        mMediaRecorder.setPreviewDisplay(surface);
        mMediaRecorder.setCamera(mCamera);
        Log.d("test","camera set");

        mMediaRecorder.setVideoSource(MediaRecorder.VideoSource.DEFAULT);
        //       mMediaRecorder.setOutputFormat(8);
        mMediaRecorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
        mMediaRecorder.setVideoEncoder(MediaRecorder.VideoEncoder.H264);
        mMediaRecorder.setVideoEncodingBitRate(512 * 1000);
        mMediaRecorder.setVideoFrameRate(30);
        mMediaRecorder.setVideoSize(1280,720);
        mMediaRecorder.setOutputFile(VIDEO_PATH_NAME);
        Log.d("test", "everything set");
        try {
            mMediaRecorder.prepare();
            Toast.makeText(getBaseContext(),
                    "media recorder prepared",
                    Toast.LENGTH_SHORT).show();
        } catch (IllegalStateException e) {
            // This is thrown if the previous calls are not called with the
            // proper order
            e.printStackTrace();
        }
        Log.d("test", "out of try catch");
        mInitSuccesful = true;
    }



    @Override
    public void surfaceDestroyed(SurfaceHolder holder)
    {
//        shutdown();
    }
    private void shutdown() {
        Log.d("test","starting shutdown");
        // Release MediaRecorder and especially the Camera as it's a shared
        // object that can be used by other applications
//        mMediaRecorder.reset();
//        mMediaRecorder.release();
        mCamera.release();

        // once the objects have been released they can't be reused
//        mMediaRecorder = null;
        mCamera = null;
    }


    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width,
                               int height)
    {
    }

    @Override
    public void onInfo(MediaRecorder mr, int what, int extra)
    {
        if(what == MediaRecorder.MEDIA_RECORDER_INFO_MAX_DURATION_REACHED)
        {
            Toast.makeText(this, "Stopping the recording due to recording limit",
                    Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public void onError(MediaRecorder mr, int what, int extra)
    {

        Toast.makeText(this, "Recording error",
                Toast.LENGTH_SHORT).show();
    }



    @SuppressWarnings("deprecation")
    private boolean initCamera() {
        try
        {
            mCamera  = Camera.open();
            mCamera.lock();
            mCamera.setDisplayOrientation(90);

            Camera.Parameters mycamparams = mCamera.getParameters();
            mycamparams.setSceneMode(Camera.Parameters.SCENE_MODE_AUTO);
            mycamparams.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
            mycamparams.setPictureSize(1280, 720); // change according to need
            mCamera.setParameters(mycamparams);

            mHolder = mVideoView.getHolder();
            mHolder.addCallback(this);
            mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
        }
        catch(RuntimeException re)
        {
            Toast.makeText(getBaseContext(),
                    "Error! Camera problem",
                    Toast.LENGTH_SHORT).show();
            re.printStackTrace();
            return false;
        }
        return true;
    }

    public void calculateAccMagOrientation()
    {
        if(SensorManager.getRotationMatrix(rotationMatrix, null, grav, magnet))
        {
            SensorManager.getOrientation(rotationMatrix, accMagOrientation);
        }
    }

    private void getRotationVectorFromGyro(float[] gyroValues,float[] deltaRotationVector,float timeFactor)
    {
        float[] normValues = new float[3];

        // Calculate the angular speed of the sample
        float omegaMagnitude =
                (float)Math.sqrt(gyroValues[0] * gyroValues[0] +
                        gyroValues[1] * gyroValues[1] +
                        gyroValues[2] * gyroValues[2]);

        // Normalize the rotation vector if it's big enough to get the axis
        if(omegaMagnitude > EPSILON)
        {
            normValues[0] = gyroValues[0] / omegaMagnitude;
            normValues[1] = gyroValues[1] / omegaMagnitude;
            normValues[2] = gyroValues[2] / omegaMagnitude;
        }

        // Integrate around this axis with the angular speed by the timestep
        // in order to get a delta rotation from this sample over the timestep
        // We will convert this axis-angle representation of the delta rotation
        // into a quaternion before turning it into the rotation matrix.
        float thetaOverTwo = omegaMagnitude * timeFactor;
        float sinThetaOverTwo = (float)Math.sin(thetaOverTwo);
        float cosThetaOverTwo = (float)Math.cos(thetaOverTwo);
        deltaRotationVector[0] = sinThetaOverTwo * normValues[0];
        deltaRotationVector[1] = sinThetaOverTwo * normValues[1];
        deltaRotationVector[2] = sinThetaOverTwo * normValues[2];
        deltaRotationVector[3] = cosThetaOverTwo;
    }

    public void gyroFunction(SensorEvent event)
    {
        // initialisation of the gyroscope based rotation matrix
        if (initState)
        {
            float[] initMatrix = new float[9];
            initMatrix = getRotationMatrixFromOrientation(accMagOrientation);
            float[] test = new float[3];
            SensorManager.getOrientation(initMatrix, test);
            gyroMatrix = matrixMultiplication(gyroMatrix, initMatrix);
            initState = false;
        }
        // copy the new gyro values into the gyro array
        // convert the raw gyro data into a rotation vector
        float[] deltaVector = new float[4];
        if (timestamp != 0)
        {
            final float dT = (event.timestamp - timestamp) * NS2S;
            System.arraycopy(event.values, 0, gyro, 0, 3);
            getRotationVectorFromGyro(gyro, deltaVector, dT / 2.0f);
        }

        // measurement done, save current time for next interval
        timestamp = event.timestamp;

        // convert rotation vector into rotation matrix
        float[] deltaMatrix = new float[9];
        SensorManager.getRotationMatrixFromVector(deltaMatrix, deltaVector);

        // apply the new rotation interval on the gyroscope based rotation matrix
        gyroMatrix = matrixMultiplication(gyroMatrix, deltaMatrix);

        // get the gyroscope based orientation from the rotation matrix
        SensorManager.getOrientation(gyroMatrix, gyroOrientation);

        float oneMinusCoeff = 1.0f - FILTER_COEFFICIENT;
            /*
             * Fix for 179° <--> -179° transition problem:
             * Check whether one of the two orientation angles (gyro or accMag) is negative while the other one is positive.
             * If so, add 360° (2 * math.PI) to the negative value, perform the sensor fusion, and remove the 360° from the result
             * if it is greater than 180°. This stabilizes the output in positive-to-negative-transition cases.
             */

        // azimuth
        if (gyroOrientation[0] < -0.5 * Math.PI && accMagOrientation[0] > 0.0) {
            fusedOrientation[0] = (float) (FILTER_COEFFICIENT * (gyroOrientation[0] + 2.0 * Math.PI) + oneMinusCoeff * accMagOrientation[0]);
            fusedOrientation[0] -= (fusedOrientation[0] > Math.PI) ? 2.0 * Math.PI : 0;
        }
        else if (accMagOrientation[0] < -0.5 * Math.PI && gyroOrientation[0] > 0.0) {
            fusedOrientation[0] = (float) (FILTER_COEFFICIENT * gyroOrientation[0] + oneMinusCoeff * (accMagOrientation[0] + 2.0 * Math.PI));
            fusedOrientation[0] -= (fusedOrientation[0] > Math.PI)? 2.0 * Math.PI : 0;
        }
        else {
            fusedOrientation[0] = FILTER_COEFFICIENT * gyroOrientation[0] + oneMinusCoeff * accMagOrientation[0];
        }

        // pitch
        if (gyroOrientation[1] < -0.5 * Math.PI && accMagOrientation[1] > 0.0) {
            fusedOrientation[1] = (float) (FILTER_COEFFICIENT * (gyroOrientation[1] + 2.0 * Math.PI) + oneMinusCoeff * accMagOrientation[1]);
            fusedOrientation[1] -= (fusedOrientation[1] > Math.PI) ? 2.0 * Math.PI : 0;
        }
        else if (accMagOrientation[1] < -0.5 * Math.PI && gyroOrientation[1] > 0.0) {
            fusedOrientation[1] = (float) (FILTER_COEFFICIENT * gyroOrientation[1] + oneMinusCoeff * (accMagOrientation[1] + 2.0 * Math.PI));
            fusedOrientation[1] -= (fusedOrientation[1] > Math.PI)? 2.0 * Math.PI : 0;
        }
        else {
            fusedOrientation[1] = FILTER_COEFFICIENT * gyroOrientation[1] + oneMinusCoeff * accMagOrientation[1];
        }

        // roll
        if (gyroOrientation[2] < -0.5 * Math.PI && accMagOrientation[2] > 0.0) {
            fusedOrientation[2] = (float) (FILTER_COEFFICIENT * (gyroOrientation[2] + 2.0 * Math.PI) + oneMinusCoeff * accMagOrientation[2]);
            fusedOrientation[2] -= (fusedOrientation[2] > Math.PI) ? 2.0 * Math.PI : 0;
        }
        else if (accMagOrientation[2] < -0.5 * Math.PI && gyroOrientation[2] > 0.0) {
            fusedOrientation[2] = (float) (FILTER_COEFFICIENT * gyroOrientation[2] + oneMinusCoeff * (accMagOrientation[2] + 2.0 * Math.PI));
            fusedOrientation[2] -= (fusedOrientation[2] > Math.PI)? 2.0 * Math.PI : 0;
        }
        else {
            fusedOrientation[2] = FILTER_COEFFICIENT * gyroOrientation[2] + oneMinusCoeff * accMagOrientation[2];
        }

        // overwrite gyro matrix and orientation with fused orientation
        // to comensate gyro drift
        gyroMatrix = getRotationMatrixFromOrientation(fusedOrientation);
        System.arraycopy(fusedOrientation, 0, gyroOrientation, 0, 3);

    }



    private float[] getRotationMatrixFromOrientation(float[] o) {
        float[] xM = new float[9];
        float[] yM = new float[9];
        float[] zM = new float[9];

        float sinX = (float)Math.sin(o[1]);
        float cosX = (float)Math.cos(o[1]);
        float sinY = (float)Math.sin(o[2]);
        float cosY = (float)Math.cos(o[2]);
        float sinZ = (float)Math.sin(o[0]);
        float cosZ = (float)Math.cos(o[0]);

        // rotation about x-axis (pitch)
        xM[0] = 1.0f; xM[1] = 0.0f; xM[2] = 0.0f;
        xM[3] = 0.0f; xM[4] = cosX; xM[5] = sinX;
        xM[6] = 0.0f; xM[7] = -sinX; xM[8] = cosX;

        // rotation about y-axis (roll)
        yM[0] = cosY; yM[1] = 0.0f; yM[2] = sinY;
        yM[3] = 0.0f; yM[4] = 1.0f; yM[5] = 0.0f;
        yM[6] = -sinY; yM[7] = 0.0f; yM[8] = cosY;

        // rotation about z-axis (azimuth)
        zM[0] = cosZ; zM[1] = sinZ; zM[2] = 0.0f;
        zM[3] = -sinZ; zM[4] = cosZ; zM[5] = 0.0f;
        zM[6] = 0.0f; zM[7] = 0.0f; zM[8] = 1.0f;

        // rotation order is y, x, z (roll, pitch, azimuth)
        float[] resultMatrix = matrixMultiplication(xM, yM);
        resultMatrix = matrixMultiplication(zM, resultMatrix);
        return resultMatrix;
    }

    public float[] matrixMultiplication(float[] A, float[] B) {
        float[] result = new float[9];

        result[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
        result[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
        result[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];

        result[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
        result[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
        result[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];

        result[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
        result[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
        result[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];

        return result;
    }




    public void ProcessData()
    {




        Log.d("test", "Starting ProcessData\n");
        ChangeAccelerationFrame(2);
        Log.d("test", "acc in ground frams\n");

        RemoveGravity(10);
        Log.d("test", "gravity removed\n");



        ChangeTimeArray();//deltat
        Log.d("test", "time modified\n");

        float stda[]=StandardDeviation(); //for image id=0
        Log.d("test", "deviation obtained"+stda[0]+" y " + stda[1] + " z " + stda[2]);

//        mess.setText("x " + stda[0] + " y " + stda[1] + " z " + stda[2]);
////        mess.setText(stda[1]);
        CorrectAcceleration(); //static bias
        Log.d("test", "static bias removed");

        temp= new double[timesum.size()][timesum.size()];
        for (int i=0; i<timesum.size(); i++)
        {
            for (int j=0; j<timesum.size(); j++)
            {
                temp[i][j]=0.0;
            }
        }
        X=new double[timev.size()][2];
        Yx=new double[timev.size()][1];
        Yy=new double[timev.size()][1];
        Yz=new double[timev.size()][1];
        for(int i=0;i<timev.size();i++)
        {
            X[i][0]=1.0;
            X[i][1]=(double)((long)timesum.get(i));
            Yx[i][0]=(double)(float)ax.get(i);
            Yy[i][0]=(double)(float)ay.get(i);
            Yz[i][0]=(double)(float)az.get(i);
        }
        tempx=new Matrix(X);
        XT=tempx.transpose();
        MatrixYx = new Matrix(Yx);
        MatrixYy = new Matrix(Yy);
        MatrixYz = new Matrix(Yz);


        SmoothAcceleration(0.05f, 4);
        Log.d("test", "Smoothened");
        boolean motionDatax[]=GetMotionPoints(5,stda,0); // 2 or 5 times stda
        boolean motionDatay[]=GetMotionPoints(5,stda,1); // 2 or 5 times stda
        boolean motionDataz[]=GetMotionPoints(5,stda,2); // 2 or 5 times stda
        Log.d("test", "Getting motion zones\n");
        Vector<int[]> mZonesx=MotionZones(motionDatax);
        Vector<int[]> mZonesy=MotionZones(motionDatay);
        Vector<int[]> mZonesz=MotionZones(motionDataz);

        for (int i=0; i< mZonesx.size() ; i++)
        {
            int[] a1= mZonesx.elementAt(i);
            Log.d("mzones x",a1[0]+" start " + a1[1]);
        }

        for (int i=0; i< mZonesy.size() ; i++)
        {
            int[] a1= mZonesy.elementAt(i);
            Log.d("mzones y",a1[0]+" start " + a1[1]);
        }

        for (int i=0; i< mZonesz.size() ; i++)
        {
            int[] a1= mZonesz.elementAt(i);
            Log.d("mzones z",a1[0]+" start " + a1[1]);
        }

        Log.d("test", "starting velocity\n");
        float velx[]=GetVelocity(mZonesx,0,2);
        float vely[]=GetVelocity(mZonesy,1,2);
        float velz[]=GetVelocity(mZonesz,2,2);

        Log.d("test", "starting distance\n");
        float dx[] = GetDistance(velx);
        float dy[] = GetDistance(vely);
        float dz[] = GetDistance(velz);

        Log.d("test","starting logging of processed data");
        OutputData(velx, vely, velz, dx, dy, dz,stda,motionDatax,motionDatay,motionDataz);
        Log.d("test", "Finishing ProcessData\n");
    }

    public void OutputData(float[] velox,float[] veloy, float[] veloz, float[] disx,float[] disy, float[] disz,float[] stda,boolean[] mx, boolean[] my, boolean[] mz)
    {
        try
        {

            fileS2 = fileDir + "/"+  curtime +  "SensorFusion3data.csv";
            BufferedWriter outdata = new BufferedWriter(new FileWriter(fileS2));
            outdata.write("Serial,Time,Imid,ax,ay,az,vx,vy,vz,dx,dy,dz,mx,my,mz,"+ stda[0] + "," + stda[1] + "," + stda[2] + "\n");

            float tottime=0;

            for(int i=0;i<ax.size();i++)
            {
                tottime+=((long) timev.get(i)*NS2S);
                outdata.write(i + "," + tottime + "," + iid.get(i) + "," + ax.get(i) + "," + ay.get(i)+ "," + az.get(i)+ "," + velox[i]+ "," + veloy[i]+ "," + veloz[i]+ "," + disx[i]+ "," + disy[i]+ "," + disz[i]+
                        "," + mx[i]+ "," + my[i]+ "," + mz[i]+
                        "\n");
                outdata.flush();
            }
            outdata.close();
        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
    }

    public float[] GetVelocity(Vector<int []> mZone, int direction,int type)
    {
        float ans[]=new float[iid.size()];

        for (int i=0; i<iid.size();i++)
        {
            ans[i]=0.0f;
        }

        for (int i=0; i< mZone.size(); i++)
        {
            float totalacc=0.0f;
            float tottime=0.0f;
            for (int j=mZone.get(i)[0]; j<mZone.get(i)[1] ; j++)
            {
                if (j>0)
                {
                    tottime+= (long) timev.get(j);
                    if (direction == 0)       // x direction
                    {
                        totalacc += (float) ax.get(j)* (long) timev.get(j);
                        ans[j] = ans[j - 1] + ((float) ax.get(j)) * ((long) timev.get(j)*NS2S);
                    } else if (direction == 1)       // y direction
                    {
                        totalacc += (float) ay.get(j)* (long) timev.get(j);
                        ans[j] = ans[j - 1] + ((float) ay.get(j)) * ((long) timev.get(j) *NS2S);
                    } else if (direction == 2)       // z direction
                    {
                        totalacc += (float) az.get(j)* (long) timev.get(j);
                        ans[j] = ans[j - 1] + ((float) az.get(j)) * ((long) timev.get(j) *NS2S);
                    }
                }
            }

            if(type==0) { // Order 0
                float correctingfact = totalacc / tottime;
                for (int j = mZone.get(i)[0]; j < mZone.get(i)[1]; j++) {
                    ans[j] -= correctingfact;
                }
            }
            else if(type==1) { // Order 1
                float finalV = ans[mZone.get(i)[1] - 1];
                float t0 = tottime * NS2S;
                long timeelapsed=0;
                float m= finalV/t0;
                for (int j = mZone.get(i)[0]; j < mZone.get(i)[1]; j++) {
                    ans[j] -= m * timeelapsed*NS2S;
                    if (j < timev.size() - 1) {
                        timeelapsed += (long) timev.get(j + 1);
                    }
                }
            }
            else if (type==2) // Order 2
            {
                float finalV = ans[mZone.get(i)[1] - 1];
                float t0 = tottime * NS2S;
                long timeelapsed=0;
                float m = 2*finalV/(t0*t0);
                for (int j = mZone.get(i)[0]; j < mZone.get(i)[1]; j++) {
                    ans[j] -= 0.5*m*timeelapsed*timeelapsed*NS2S*NS2S;
                    if (j < timev.size() - 1) {
                        timeelapsed += (long) timev.get(j + 1);
                    }
                }
            }
        }

        return ans;
    }

    public float[] GetDistance(float velarray[])
    {
        float answer[] = new float[iid.size()];
        answer[0]=0.0f;

        for (int i=1; i<iid.size() ; i++)
        {
            answer[i]=answer[i-1]+ (velarray[i]*((long) timev.get(i))*NS2S);
        }

        return answer;
    }

    public Vector<int[]> MotionZones(boolean[] data)
    {
        Vector<int[]> ans=new Vector<int[]>();

        boolean state=false;
        int previousval=0;
        for(int i=0;i<timev.size();i++)
        {
           if (data[i])
           {
               if (state)
               {
                    int p=2+2;
               }
               else
               {
                   Log.d("mzones","start at " + i);
                   state=true;
                   previousval = i;
               }
           }
           else
           {
               if (state)
               {
                   Log.d("mzones","stop at " + i);
                   state=false;
                   int temp[]=new int[2];
                   temp[0]=previousval;
                   temp[1]=i;
                   ans.addElement(temp);
               }
               else
               {
                    int q=5;
               }
           }
        }
        if(state)
        {
            Log.d("mzones","stop at " + (ax.size()-1));
            state=false;
            int temp[]=new int[2];
            temp[0]=previousval;
            temp[1]=ax.size()-1;
            ans.addElement(temp);
        }
        return ans;
    }

    public void ChangeAccelerationFrame(int typeofmatrix)
    {
        if (typeofmatrix==0)
        {
//            Use RotationMatrix
            for (int i=0; i<ax.size(); i++)
            {
                float temp1=((float) ax.get(i))*(rotmat1.get(i)[0])  + ((float) ay.get(i))*(rotmat1.get(i)[1]) + ((float) az.get(i))*(rotmat1.get(i)[2]);
                float temp2=((float) ax.get(i))*(rotmat1.get(i)[3])  + ((float) ay.get(i))*(rotmat1.get(i)[4]) + ((float) az.get(i))*(rotmat1.get(i)[5]);
                float temp3=((float) ax.get(i))*(rotmat1.get(i)[6])  + ((float) ay.get(i))*(rotmat1.get(i)[7]) + ((float) az.get(i))*(rotmat1.get(i)[8]);
                ax.setElementAt(temp1,i);
                ay.setElementAt(temp2,i);
                az.setElementAt(temp3,i);
            }
        }
        else if (typeofmatrix==1)
        {
//            Use GyroMatrix
            for (int i=0; i<ax.size(); i++)
            {
                float temp1=((float) ax.get(i))*(rotmat2.get(i)[0])  + ((float) ay.get(i))*(rotmat2.get(i)[1]) + ((float) az.get(i))*(rotmat2.get(i)[2]);
                float temp2=((float) ax.get(i))*(rotmat2.get(i)[3])  + ((float) ay.get(i))*(rotmat2.get(i)[4]) + ((float) az.get(i))*(rotmat2.get(i)[5]);
                float temp3=((float) ax.get(i))*(rotmat2.get(i)[6])  + ((float) ay.get(i))*(rotmat2.get(i)[7]) + ((float) az.get(i))*(rotmat2.get(i)[8]);
                ax.setElementAt(temp1,i);
                ay.setElementAt(temp2,i);
                az.setElementAt(temp3,i);
            }
        }
        else if (typeofmatrix==2)
        {
            Log.d("motion","IN CORRECT ELSE IF CONDITION");

//            Using rotmatrix ax-gx
            for (int i=0; i<ax.size(); i++)
            {
                float temp1=(((float) ax.get(i) - (float) gx.get(i)) * (rotmat1.get(i)[0])) + (((float) ay.get(i) - (float) gy.get(i)) * (rotmat1.get(i)[1])) + (((float) az.get(i) - (float) gz.get(i)) * (rotmat1.get(i)[2]));
                float temp2=(((float) ax.get(i) - (float) gx.get(i)) * (rotmat1.get(i)[3])) + (((float) ay.get(i) - (float) gy.get(i)) * (rotmat1.get(i)[4])) + (((float) az.get(i) - (float) gz.get(i)) * (rotmat1.get(i)[5]));
                float temp3=(((float) ax.get(i) - (float) gx.get(i)) * (rotmat1.get(i)[6])) + (((float) ay.get(i) - (float) gy.get(i)) * (rotmat1.get(i)[7])) + (((float) az.get(i) - (float) gz.get(i)) * (rotmat1.get(i)[8]));
//                Log.d("motion", "before: x" + ax.get(i) + " y: " + ay.get(i) + " z: " + az.get(i) + " gx: " + gx.get(i)+ " gy: " + gy.get(i) + " gz: "+ gz.get(i)+ " rot7: "+rotmat1.get(i)[6]+ " rot8: "+rotmat1.get(i)[7]+ " rot9: "+rotmat1.get(i)[8]+ " settingx "+temp1+ " settingy "+temp2+ " settingz "+temp3);
                ax.setElementAt(temp1, i);
                ay.setElementAt(temp2, i);
                az.setElementAt(temp3,i);
//                Log.d("motion", "after: x" + ax.get(i) + " y: " + ay.get(i) + " z: " + az.get(i));
            }

        }
    }

    public float[] StandardDeviation()
    {
        float result[]=new float[3];
        result[0]=0.0f;
        result[1]=0.0f;
        result[2]=0.0f;
        float mean1=0.0f;
        float mean2=0.0f;
        float mean3=0.0f;
        int total=0;

        for(int i=0;i<iid.size();i++)
        {
            if((int)iid.get(i)==0)
            {
                total++;
            }
            else
            {
                break;
            }
        }

        for(int i=0;i<total;i++)
        {
            mean1 = mean1 + (float)(ax.get(i));
            mean2 = mean2 + (float)(ay.get(i));
            mean3 = mean3 + (float)(az.get(i));
        }
        mean1=mean1/total;
        mean2=mean2/total;
        mean3=mean3/total;

        Log.d("mzone stda static bias","x: "+ mean1 + " y:"+ mean2 +"z:"+mean3);

        for(int i=0;i<total;i++)
        {
            result[0]+= Math.pow((float)(ax.get(i))-mean1,2.0);
            result[1]+= Math.pow((float)(ay.get(i))-mean2,2.0);
            result[2]+= Math.pow((float)(az.get(i))-mean3,2.0);
        }
        result[0]=(float) Math.sqrt(result[0]/(total));
        result[1]=(float) Math.sqrt(result[1]/(total));
        result[2]=(float) Math.sqrt(result[2]/(total));
        return result;
    }

    public void RemoveGravity(int typeremoval)
    {
//        0 for removal of 9.8, 1 for removal of gx,gy,gz using RotationMatrix, 2 for removal of gx,gy,gz using GyroMatrix
        if (typeremoval==0)
        {
            for (int i=0; i<az.size();i++)
            {
                float temp=((float) az.get(i))-9.806f;
                az.setElementAt(temp,i);
            }
        }
        else if (typeremoval==1)
        {
            for (int i=0; i<ax.size(); i++)
            {
                float temp1= ((float) ax.get(i)) - ((float) gx.get(i))*(rotmat1.get(i)[0])  - ((float) gy.get(i))*(rotmat1.get(i)[1]) - ((float) gz.get(i))*(rotmat1.get(i)[2]);
                float temp2= ((float) ay.get(i)) - ((float) gx.get(i))*(rotmat1.get(i)[3])  - ((float) gy.get(i))*(rotmat1.get(i)[4]) - ((float) gz.get(i))*(rotmat1.get(i)[5]);
                float temp3= ((float) ay.get(i)) - ((float) gx.get(i))*(rotmat1.get(i)[6])  - ((float) gy.get(i))*(rotmat1.get(i)[7]) - ((float) gz.get(i))*(rotmat1.get(i)[8]);
                ax.setElementAt(temp1,i);
                ay.setElementAt(temp2,i);
                az.setElementAt(temp3,i);
            }
        }
        else if (typeremoval==2)
        {
            for (int i=0; i<ax.size(); i++)
            {
                float temp1=((float) ax.get(i)) - ((float) gx.get(i))*(rotmat2.get(i)[0])  - ((float) gy.get(i))*(rotmat2.get(i)[1]) - ((float) gz.get(i))*(rotmat2.get(i)[2]);
                float temp2=((float) ax.get(i)) - ((float) gx.get(i))*(rotmat2.get(i)[3])  - ((float) gy.get(i))*(rotmat2.get(i)[4]) - ((float) gz.get(i))*(rotmat2.get(i)[5]);
                float temp3=((float) az.get(i)) - ((float) gx.get(i))*(rotmat2.get(i)[6])  - ((float) gy.get(i))*(rotmat2.get(i)[7]) - ((float) gz.get(i))*(rotmat2.get(i)[8]);
                ax.setElementAt(temp1,i);
                ay.setElementAt(temp2,i);
                az.setElementAt(temp3,i);
            }
        }
    }

    public void ChangeTimeArray()
    {
        long prevtime=0;
        for (int i=0; i<timev.size(); i++)
        {
            long prevtime2=(long) timev.get(i);
            long temp=(long) timev.get(i)-prevtime;
            timev.setElementAt(temp,i);
            prevtime=prevtime2;
        }
    }

    public void CorrectAcceleration()
    {
        int total=0;
        float totacc[]=new float[3];

        totacc[0]=0.0f;
        totacc[1]=0.0f;
        totacc[2]=0.0f;



        Log.d("kest", "starting while loop in sb" + iid.size()+ "ax:" + ax.size()+"ay:"+ay.size()+"az:"+az.size());

        while ((total<iid.size()))
        {
            if (((int)iid.get(total))==0)
            {
//                Log.d("test", "In while" + total);
                totacc[0] += (float) ax.get(total);
                totacc[1] += (float) ay.get(total);
                totacc[2] += (float) az.get(total);
                total += 1;
//                Log.d("test", "out while" + total);
            }
            else
            {
                break;
            }
        }

        Log.d("mzone test", "in static  bias corr"+ total);


        totacc[0] /= total;
        totacc[1] /= total;
        totacc[2] /= total;

        Log.d("mzone test", "static bias obtained " + totacc[0]+ " y "+totacc[1]+" z "+totacc[2]);

        for (int i=0; i<ax.size(); i++)
        {
            float ax1 = (float) ax.get(i) - totacc[0];
            float ay1 = (float) ay.get(i) - totacc[1];
            float az1 = (float) az.get(i) - totacc[2];
            ax.setElementAt( ax1 , i);
            ay.setElementAt( ay1 , i);
            az.setElementAt( az1 , i);
        }
    }

    public void SmoothAcceleration(float alfa,int type)
    {
        if(type==0) {
            //low pass filter with alfa
            for (int i = 1; i < ax.size(); i++) {
                float temp = ((float) ax.get(i)) * alfa + (1 - alfa) * ((float) ax.get(i - 1));
                ax.setElementAt(temp, i);
                temp = ((float) ay.get(i)) * alfa + (1 - alfa) * ((float) ay.get(i - 1));
                ay.setElementAt(temp, i);
                temp = ((float) az.get(i)) * alfa + (1 - alfa) * ((float) az.get(i - 1));
                az.setElementAt(temp, i);
            }
        }
        //Gaussian Filter
        else if(type==1) {
            for (int i = 0; i < ax.size(); i++) {

            }
        }
        else if(type==2){
            for(int i=0;i<timev.size();i++)
            {
                Log.d("loess","on index" + i + "out of " + timev.size());
                loess_query((long) timesum.get(i), alfa, i);
            }
        }
        else if(type==4){
            for (int i=0; i<timev.size();i++)
            {
                DoEveryThing(i,alfa);
            }
        }
    }

    public float W(int i,int m,int n,float alpha) {
        return (float) Math.pow((1 - Math.pow((2.0 * (Math.abs(i - m))) / (n * alpha),3)),3);
    }

    public void DoEveryThing(int m, float alpha) {
        int n = timev.size();
        float sigwi =0.0f;
        float sigiwi =0.0f;
        float sigi2wi =0.0f;
        float sigxiwi =0.0f;
        float sigixiwi = 0.0f;
        float sigyiwi =0.0f;
        float sigiyiwi = 0.0f;
        float sigziwi =0.0f;
        float sigiziwi = 0.0f;
        int nalphaby2 =(int) (n * alpha / 2);
        int i = m - nalphaby2;
        int finish = m + nalphaby2;
        if (i< 0)
        {
            i = 0;
        }
        if (finish > n) {
            finish = n;
        }
        while (i < finish)
        {
            float wi = W(i, m, n, alpha);
            sigwi += wi;
            sigiwi += i * wi;
            sigi2wi += i * i * wi;
            float mulx= (float) (((float) ax.get(i))*wi);
            sigxiwi += mulx;
            sigixiwi += i * mulx;
            sigyiwi += ((float) ay.get(i)) * wi;
            sigiyiwi += i * ((float) ay.get(i)) * wi;
            sigziwi += ((float) az.get(i)) * wi;
            sigiziwi += i * ((float) az.get(i)) * wi;
            i += 1;
        }
        float denom = ((sigwi) * (sigi2wi)) - (float) Math.pow((sigiwi),2);

        float numerx = (sigi2wi) * (sigxiwi) + m * (sigwi) * (sigixiwi) ;
        numerx -= (sigiwi) * (sigixiwi) ;
        numerx -= m * sigxiwi * sigiwi ;

        float numery = (sigi2wi) * (sigyiwi) + m * (sigwi) * (sigiyiwi) ;
        numery -= (sigiwi) * (sigiyiwi) ;
        numery -= m * sigyiwi * sigiwi ;

        float numerz = (sigi2wi) * (sigziwi) + m * (sigwi) * (sigiziwi) ;
        numerz -= (sigiwi) * (sigiziwi) ;
        numerz -= m * sigziwi * sigiwi ;

        ax.setElementAt((float) numerx/denom,m);
        ay.setElementAt((float) numery/denom,m);
        az.setElementAt((float) numerz/denom,m);

    }
    public void loess_query(long time, float alfa,int post)
    {
        long p= (new Date()).getTime();
        Matrix W=weigths_matrix(time,alfa,post);
        Log.d("weight matrix obtained","time taken " + ((new Date()).getTime() -p) );
//        for(int i=0;i<timev.size();i++)
//        {
//            X[i][0]=1.0;
//            X[i][1]=(double)((long)timesum.get(i));
//            Yx[i][0]=(double)(float)ax.get(i);
//            Yy[i][0]=(double)(float)ay.get(i);
//            Yz[i][0]=(double)(float)az.get(i);
//        }
        Matrix tempyx,tempyy,tempyz;
        Matrix temp1= XT.times(W);
        Matrix temp2= temp1.times(tempx);
        temp2=temp2.inverse();
        temp2 = temp2.times(temp1);
        tempyx=temp2.times(MatrixYx);
        tempyy=temp2.times(MatrixYy);
        tempyz=temp2.times(MatrixYz);
        double corx= tempyx.get(0,0) + time*tempyx.get(1,0);
        double cory= tempyy.get(0,0) + time*tempyy.get(1,0);
        double corz= tempyz.get(0,0) + time*tempyz.get(1,0);
        ax.setElementAt((float)corx,post);
        ay.setElementAt((float)cory,post);
        az.setElementAt((float)corz,post);
        MatrixYx.set(post,0,corx);
        MatrixYy.set(post,0,cory);
        MatrixYz.set(post,0,corz);
        Log.d("weight matrix obtained", "time taken " + ((new Date()).getTime() - p));
    }

    public Matrix weigths_matrix(long x,float alfa,int post)
    {
        long tstart= (new Date()).getTime();

        int lim=(int)(timesum.size()*alfa);
        for(int i=0;i<timesum.size();i++)
        {
            temp[i][i]=0.0;
//            for(int j=0;j<timesum.size();j++) {
//                temp[i][j] = 0.0;
//            }
        }
        int temp1 = post-lim/2;
        int temp2 = post+lim/2;
        double normfactor;
        if(temp2<timesum.size())
        {
            normfactor= Math.abs((double) (x - (long) timesum.get(temp2)));
        }
        else
        {
            normfactor= Math.abs((double) (x - (long) timesum.get(temp1)));
        }

        for(int i=temp1;i<temp2;i++) {
            if (i<0)
            {
                i=-1;
            }
            else if (i>=timesum.size())
            {
                break;
            }
            else
            {
                temp[i][i] = Math.abs((double) (x - (long) timesum.get(i)));
                temp[i][i] /= normfactor;
                double temp3= temp[i][i]*temp[i][i]*temp[i][i];
                temp[i][i]=1-temp3;
                temp3= temp[i][i]*temp[i][i]*temp[i][i];
                temp[i][i]=temp3;
            }
        }

        Log.d("loess","weights finsihed: " + ((new Date()).getTime()-tstart));
//        for(int i=0;i<timesum.size();i++)
//        {
//            temp[i][i] /=normfactor;
//            if(temp[i][i]>1)
//            {
//                temp[i][i]=0.0;
//            }
//            else
//            {
//                double temp3= temp[i][i]*temp[i][i]*temp[i][i];
//                temp[i][i]=1-temp3;
//                temp3= temp[i][i]*temp[i][i]*temp[i][i];
//                temp[i][i]=temp3;
//            }
//        }
        return new Matrix(temp);
    }

    public boolean CheckMaxRange(Vector considered,int windowssize,int index, float stdval)
    {
//        Return true is moving point
        float max1=0.0f;
        if (index<windowssize)
        {
            for (int j=0; j<index+windowssize && j<ax.size() ;j++)
            {
                max1=Math.max(Math.abs((float) considered.get(j)),max1);
            }
        }
        else if (index+windowssize>ax.size())
        {
            for (int j=index - windowssize; j<ax.size() ;j++)
            {
                max1=Math.max(Math.abs((float) considered.get(j)),max1);
            }
        }
        else
        {
            for (int j=index-windowssize; j<index+windowssize; j++)
            {
                max1=Math.max(Math.abs((float) considered.get(j)),max1);
            }
        }
        return (max1>stdval);
    }

    public boolean[] GetMotionPoints(int factor,float[] standarddev, int direction)
    {
        boolean answer[]=new boolean[ax.size()];
        if (direction==0) // x axis
        {
            for (int k=0; k<ax.size(); k++)
            {
                if((int)iid.get(k)==0)
                {
                    answer[k] = false;
                }
                else
                {
                    answer[k] = CheckMaxRange(ax, 50, k, factor * standarddev[0]);
                    if (k > 0) {
                        if ((int) iid.get(k) != (int) iid.get(k - 1)) {
                            answer[k - 1] = false;
                            answer[k] = false;
                        }
                    }
                }
            }
        }
        else if (direction==1) // y axis
        {
            for (int k=0; k<ax.size(); k++)
            {
                if((int)iid.get(k)==0)
                {
                    answer[k] = false;
                }
                else {
                    answer[k] = CheckMaxRange(ay, 50, k, factor * standarddev[1]);
                    if (k > 0) {
                        if ((int) iid.get(k) != (int) iid.get(k - 1)) {
                            answer[k - 1] = false;
                            answer[k] = false;
                        }
                    }
                }
            }
        }
        else if (direction==2) // z axis
        {
            for (int k=0; k<ax.size(); k++)
            {
                if((int)iid.get(k)==0)
                {
                    answer[k] = false;
                }
                else {
                    answer[k] = CheckMaxRange(az, 50, k, factor * standarddev[2]);
                    if (k > 0) {
                        if ((int) iid.get(k) != (int) iid.get(k - 1)) {
                            answer[k - 1] = false;
                            answer[k] = false;
                        }
                    }
                }
            }
        }
        return answer;
    }

    @Override
    public void onPause()

    {
		/*if (mCamera != null) {
	        mCamera.release();
	        mCamera = null;
	      }
	      */
        super.onPause();
        mSensorManager.unregisterListener(this);

    }

    @Override
    public void onResume()
    {
        super.onResume();
    }

    @Override
    public void onStop() {
        super.onStop();
        // unregister sensor listeners to prevent the activity from draining the device's battery.
        mSensorManager.unregisterListener(this);

    }


    @Override
    public void onDestroy()
    {

        // Unregister the listeners for all the sensors here
        mSensorManager.unregisterListener(this);
        if(mCamera != null) {
            try {
                mCamera.reconnect();
            } catch (IOException e) {
                e.printStackTrace();
            }
            mCamera.release();
            mCamera = null;
        }
        super.onDestroy();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_main, menu);
        return true;
    }
}

class FocusCallback implements Camera.AutoFocusCallback {


    private PhotoHandler myHandler;
    private Camera myCamera;

    public FocusCallback(Camera newCamera) {
        myCamera = newCamera;

    }

    public void preparePicture(PhotoHandler photoHandler) {
        myHandler = photoHandler;

    }

    @Override
    public void onAutoFocus(boolean success, Camera camera) {
        //	Log.i("Proceed: ", "Autofocus done");
        myCamera.takePicture(null, null, myHandler);

    }

}