import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState, useRef, useEffect } from 'react';
import { StyleSheet, Text, View, Dimensions } from 'react-native';
import { Audio } from 'expo-av';

const { width } = Dimensions.get('window');

export default function CameraScanner() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  
  const [navigationData, setNavigationData] = useState({
    status: "CONNECTING...",
    score: 0.0,
    objects: [] as string[]
  });

  const TAILSCALE_IP = "100.90.17.72"; 
  const SERVER_URL = `http://${TAILSCALE_IP}:8000/detect`;

  // --- 1. DEFINE AUDIO FUNCTION FIRST ---
  // This must be defined before the useEffect that calls it
  const playVoiceCommand = async (base64Audio: string) => {
    try {
      const { sound } = await Audio.Sound.createAsync(
        { uri: `data:audio/mp3;base64,${base64Audio}` },
        { shouldPlay: true }
      );

      // Cleanup memory once the speech is done
      sound.setOnPlaybackStatusUpdate((status) => {
        if (status.isLoaded && status.didJustFinish) {
          sound.unloadAsync();
        }
      });
    } catch (e) {
      console.error("Audio Playback Error:", e);
    }
  };

  // Set audio mode once on mount
  useEffect(() => {
    Audio.setAudioModeAsync({
      playsInSilentModeIOS: true,
      allowsRecordingIOS: false,
      staysActiveInBackground: true,
    });
  }, []);

  // --- 2. THE BROADCAST LOOP ---
  useEffect(() => {
    const broadcastInterval = setInterval(async () => {
      // Check if we are already processing a frame to prevent "stacking" requests
      if (cameraRef.current && !isProcessing && permission?.granted) {
        setIsProcessing(true);
        try {
          const photo = await cameraRef.current.takePictureAsync({
            base64: true,
            quality: 0.2, // Keep quality low for fast Tailscale transfer
            skipProcessing: true,
          });

          if (!photo?.base64) {
            setIsProcessing(false);
            return;
          }

          const response = await fetch(SERVER_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: photo.base64 }),
          });

          const data = await response.json();

          // Update HUD
          setNavigationData({
            status: data.navigation.status,
            score: data.navigation.danger_score,
            objects: data.objects ? data.objects.map((o: any) => o.label) : []
          });

          // Play Gemini Voice Advice
          if (data.audio && typeof data.audio === 'string') {
            await playVoiceCommand(data.audio);
          }

        } catch (e) {
          console.error("Network Error (Check Tailscale):", e);
        } finally {
          setIsProcessing(false);
        }
      }
    }, 400); // 400ms is a sweet spot for "real-time" feel without lagging the network

    return () => clearInterval(broadcastInterval);
  }, [permission, isProcessing]);

  if (!permission?.granted) {
    return (
      <View style={styles.container}>
        <Text style={styles.permissionText} onPress={requestPermission}>
          Tap to Grant Camera Permission
        </Text>
      </View>
    );
  }

  const getDangerColor = () => {
    if (navigationData.score > 0.8) return '#FF3B30'; // Red
    if (navigationData.score > 0.5) return '#FF9500'; // Orange
    return '#34C759'; // Green
  };

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} ref={cameraRef} />

      <View style={[styles.hud, { backgroundColor: getDangerColor() }]}>
        <Text style={styles.hudStatus}>{navigationData.status}</Text>
        <Text style={styles.hudScore}>Danger Score: {navigationData.score.toFixed(2)}</Text>
        
        {navigationData.objects.length > 0 && (
          <View style={styles.objectContainer}>
            <Text style={styles.objectText}>
              Seeing: {navigationData.objects.join(', ')}
            </Text>
          </View>
        )}
      </View>

      {/* Blue dot blinks when an image is actively being sent to your laptop */}
      {isProcessing && <View style={styles.blinkDot} />}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  camera: { flex: 1 },
  hud: {
    position: 'absolute',
    top: 50,
    left: 15,
    right: 15,
    padding: 20,
    borderRadius: 25,
    alignItems: 'center',
    elevation: 15,
  },
  hudStatus: { color: '#fff', fontSize: 32, fontWeight: '900', letterSpacing: 1 },
  hudScore: { color: '#fff', fontSize: 16, marginTop: 5, opacity: 0.9 },
  objectContainer: { 
    marginTop: 15, 
    borderTopWidth: 1, 
    borderTopColor: 'rgba(255,255,255,0.3)', 
    paddingTop: 10, 
    width: '100%' 
  },
  objectText: { color: '#fff', fontSize: 14, fontWeight: 'bold', textAlign: 'center' },
  permissionText: { color: '#fff', fontSize: 18, alignSelf: 'center', marginTop: '50%' },
  blinkDot: { 
    position: 'absolute', 
    bottom: 30, 
    right: 30, 
    width: 12, 
    height: 12, 
    borderRadius: 6, 
    backgroundColor: '#007AFF' 
  }
});