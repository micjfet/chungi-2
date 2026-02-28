import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState, useRef } from 'react';
import { Button, StyleSheet, Text, TouchableOpacity, View } from 'react-native';

export default function CameraScanner() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const [isScanning, setIsScanning] = useState(false);

  if (!permission) {
    // Camera permissions are still loading.
    return <View />;
  }

  if (!permission.granted) {
    // Camera permissions are not granted yet.
    return (
      <View style={styles.container}>
        <Text style={{ textAlign: 'center', marginBottom: 20 }}>
          We need your permission to show the camera for obstacle detection.
        </Text>
        <Button onPress={requestPermission} title="Grant Permission" />
      </View>
    );
  }

  const handleIdentifySurroundings = async () => {
    if (cameraRef.current && !isScanning) {
      setIsScanning(true);
      try {
        // 1. Capture the image (Optimized for Gemini 1.5/2.0 Flash)
        const photo = await cameraRef.current.takePictureAsync({
          base64: true,
          quality: 0.7, // Balance between clarity and upload speed
        });

        console.log("Image captured! Base64 length:", photo?.base64?.length);

        // 2. PLACEHOLDER: This is where you call your Gemini/ElevenLabs proxy
        // const response = await fetch('YOUR_NGROK_URL/analyze', { ... });
        
        alert("Image captured! Ready to send to Gemini.");
      } catch (e) {
        console.error("Failed to take picture:", e);
      } finally {
        setIsScanning(false);
      }
    }
  };

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} ref={cameraRef}>
        <View style={styles.buttonContainer}>
          <TouchableOpacity 
            style={[styles.button, isScanning && { backgroundColor: 'gray' }]} 
            onPress={handleIdentifySurroundings}
            disabled={isScanning}
          >
            <Text style={styles.text}>{isScanning ? "Analyzing..." : "Identify Surroundings"}</Text>
          </TouchableOpacity>
        </View>
      </CameraView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: 'transparent',
    margin: 64,
  },
  button: {
    flex: 1,
    alignSelf: 'flex-end',
    alignItems: 'center',
    backgroundColor: '#6200ee',
    padding: 20,
    borderRadius: 15,
    elevation: 5,
  },
  text: {
    fontSize: 18,
    fontWeight: 'bold',
    color: 'white',
  },
});