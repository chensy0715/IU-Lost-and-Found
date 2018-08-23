package net.xinerz.iulostandfound;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;

import org.json.JSONObject;

public class MainActivity extends AppCompatActivity implements View.OnClickListener {
    private static final int RESULT_LOAD_IMAGE = 1;
    String foundItem = null;
    ImageView imageToUpload;
    Button bUploadImage;
    EditText uploadImageName;
    TextView textView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageToUpload = (ImageView) findViewById(R.id.imageToUpload);
        bUploadImage = (Button) findViewById(R.id.bUploadImage);
        uploadImageName = (EditText) findViewById(R.id.etUploadNames);
        textView = (TextView) findViewById(R.id.textView);
        imageToUpload.setOnClickListener(this);
        bUploadImage.setOnClickListener(this);
        textView.setText("Tell us what you lost");
    }

    @Override
    public void onClick(View v) {
        switch(v.getId()) {
            case R.id.imageToUpload:
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, RESULT_LOAD_IMAGE);
                break;
            case R.id.bUploadImage:
                foundItem = uploadImageName.getText().toString();
                textView.setText(foundItem);
                View view = this.getCurrentFocus();
                InputMethodManager imm = (InputMethodManager)getSystemService(Context.INPUT_METHOD_SERVICE);
                imm.hideSoftInputFromWindow(view.getWindowToken(), 0);
                findingItem(foundItem);
                break;
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == RESULT_LOAD_IMAGE && resultCode == RESULT_OK && data != null){
            Uri selectedImage = data.getData();
            imageToUpload.setImageURI(selectedImage);
        }
    }

    protected void findingItem(String foundItem){
        NetworkActivity act = new NetworkActivity();
        act.execute();
        if (act.found) {
            StringBuilder sb = new StringBuilder();
            for(String l : act.locations) {
                sb.append(" " + l);
            }
            textView.setText("Your lost item is found in" + sb.toString());
        } else {
            textView.setText("Your " + foundItem +" is not found anywhere in the campus");
        }
    }
}