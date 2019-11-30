package com.example.realtimeobjectrecognitionmlkitcustommodel

import android.content.Intent
import android.media.MediaPlayer
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import com.google.firebase.ml.common.FirebaseMLException
import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager
import com.google.firebase.ml.common.modeldownload.FirebaseRemoteModel
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel

class SplashActivity : AppCompatActivity() {

    private val HOSTED_MODEL_NAME = "custom-model"
    private val TAG = "SplashActivity"
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_splash)
        initFirebase()
    }

    private fun initFirebase() {
        try {
            val conditions = FirebaseModelDownloadConditions.Builder().build()
            // .requireWifi().build()
            val remoteModel = FirebaseCustomRemoteModel.Builder(HOSTED_MODEL_NAME).build()
            FirebaseModelManager.getInstance().download(remoteModel, conditions)
                .addOnSuccessListener {
                    Toast.makeText(this, "Başarılı", Toast.LENGTH_SHORT).show()
                    val intent = Intent(this, RecognitionActivity::class.java)
                    startActivity(intent)
                    finish()
                }
                .addOnFailureListener { error ->

                    if (error.cause.toString().equals("com.google.firebase.ml.common.FirebaseMLException: Failed to get model URL")) {
                        Toast.makeText(
                            this,
                            "İnternet Bağlantınızı Kontrol Edin!",
                            Toast.LENGTH_SHORT
                        )
                            .show()
                    }
                    Log.e(TAG, error.message)
                }
        } catch (e: FirebaseMLException) {
            Log.e(TAG, "Error while setting up the model", e)
        }
    }
}
