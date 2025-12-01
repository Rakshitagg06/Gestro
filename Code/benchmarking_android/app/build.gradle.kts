plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.benchmarking"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.example.benchmarking"
        minSdk = 24
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        compose = false
    }
}

dependencies {
    // Android basics
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")

    // TensorFlow Lite
    implementation("org.tensorflow:tensorflow-lite:2.14.0")

    implementation("org.tensorflow:tensorflow-lite-gpu:2.14.0")

    implementation("org.tensorflow:tensorflow-lite-gpu-delegate-plugin:0.4.3")
    // JSON
    implementation("com.google.code.gson:gson:2.10.1")

    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
        implementation ("org.tensorflow:tensorflow-lite:2.14.0")
        implementation ("org.tensorflow:tensorflow-lite-gpu:2.14.0")
        // Optional: include if your model uses Select TF ops
        // implementation "org.tensorflow:tensorflow-lite-select-tf-ops:2.14.0"

        // MediaPipe Tasks Vision
        implementation ("com.google.mediapipe:tasks-vision:0.20230731") // or latest stable

        // JSON
        implementation ("com.google.code.gson:gson:2.10.1")

        // DocumentFile
        implementation ("androidx.documentfile:documentfile:1.0.1")
}
