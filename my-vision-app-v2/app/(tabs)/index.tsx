import { CameraView, useCameraPermissions } from 'expo-camera';
import { useState, useRef, useEffect } from 'react';
import { StyleSheet, Text, View, Dimensions, Vibration, Pressable, ActionSheetIOS, Platform, Alert } from 'react-native';
import { Audio } from 'expo-av';
import { SymbolView } from 'expo-symbols';

const { width } = Dimensions.get('window');

const labels: Record<string, string> = {
  en: "Danger",
  es: "Peligro",
  fr: "Danger",
  vi: "Mức độ nguy hiểm",
  ja: "危険度",
  zh: "危险指数"
};

export default function CameraScanner() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef<CameraView>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState("en");
  
  // Throttle ref to prevent vibration spam
  const lastVibrationTime = useRef<number>(0);
  const VIBRATION_THROTTLE_MS = 1500; 

  const [navigationData, setNavigationData] = useState({
    status: "CONNECTING...",
    score: 0.0,
    objects: [] as string[]
  });

  // --- NETWORKING CONFIG ---
  const TAILSCALE_IP = "100.90.17.72"; 
  const SERVER_URL = `http://${TAILSCALE_IP}:8000/detect`;
  const LANG_URL = `http://${TAILSCALE_IP}:8000/language`;

  // --- LANGUAGE SELECTION SETUP ---
  const languages = [
    { label: 'English', code: 'en' },
    { label: 'Español', code: 'es' },
    { label: 'Français', code: 'fr' },
    { label: 'Tiếng Việt', code: 'vi' },
    { label: '日本語', code: 'ja' },
    { label: '中文', code: 'zh' },
    { label: 'Cancel', code: 'cancel' },
  ];

  const changeLanguage = async (code: string) => {
    if (code === 'cancel') return;
    try {
      const response = await fetch(LANG_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ lang: code }),
      });
      if (response.ok) {
        setSelectedLanguage(code);
        Vibration.vibrate(50); // Haptic feedback for selection
      }
    } catch (e) {
      console.error("Failed to sync language with server:", e);
    }
  };

  const showLanguageMenu = () => {
    if (Platform.OS === 'ios') {
      ActionSheetIOS.showActionSheetWithOptions(
        {
          options: languages.map(l => l.label),
          cancelButtonIndex: 6,
          userInterfaceStyle: 'dark',
        },
        (buttonIndex) => changeLanguage(languages[buttonIndex].code)
      );
    } else {
      Alert.alert(
        "Select Language",
        "Choose voice guidance language:",
        languages.filter(l => l.code !== 'cancel').map(l => ({
          text: l.label,
          onPress: () => changeLanguage(l.code)
        }))
      );
    }
  };

  // --- AUDIO LOGIC ---
  const playVoiceCommand = async (base64Audio: string) => {
    try {
      const { sound } = await Audio.Sound.createAsync(
        { uri: `data:audio/mp3;base64,${base64Audio}` },
        { shouldPlay: true }
      );
      sound.setOnPlaybackStatusUpdate((status) => {
        if (status.isLoaded && status.didJustFinish) {
          sound.unloadAsync();
        }
      });
    } catch (e) {
      console.error("Audio Playback Error:", e);
    }
  };

  useEffect(() => {
    Audio.setAudioModeAsync({
      playsInSilentModeIOS: true,
      allowsRecordingIOS: false,
      staysActiveInBackground: true,
    });
  }, []);

  // --- CORE BROADCAST LOOP ---
  useEffect(() => {
    const broadcastInterval = setInterval(async () => {
      if (cameraRef.current && !isProcessing && permission?.granted) {
        setIsProcessing(true);
        try {
          const photo = await cameraRef.current.takePictureAsync({
            base64: true,
            quality: 0.2,
            skipProcessing: true,
          });

          if (!photo?.base64) {
            setIsProcessing(false);
            return;
          }

          const response = await fetch(SERVER_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
              image: photo.base64,
              lang: selectedLanguage // Sends language choice to server
            }),
          });

          const data = await response.json();
          const dangerScore = data.navigation.danger_score;

          // Update HUD
          setNavigationData({
            status: data.navigation.status,
            score: dangerScore,
            objects: data.objects ? data.objects.map((o: any) => o.label) : []
          });

          // Vibration Logic
          const now = Date.now();
          if (now - lastVibrationTime.current > VIBRATION_THROTTLE_MS) {
            if (dangerScore > 0.8) {
              Vibration.vibrate([0, 400, 100, 400]);
              lastVibrationTime.current = now;
            } else if (dangerScore > 0.5) {
              Vibration.vibrate(200);
              lastVibrationTime.current = now;
            }
          }

          if (data.audio) {
            await playVoiceCommand(data.audio);
          }

        } catch (e) {
          console.error("Inference Error:", e);
        } finally {
          setIsProcessing(false);
        }
      }
    }, 400);

    return () => clearInterval(broadcastInterval);
  }, [permission, isProcessing, selectedLanguage]);

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
    if (navigationData.score > 0.8) return '#FF3B30'; 
    if (navigationData.score > 0.5) return '#FF9500'; 
    return '#34C759'; 
  };

  return (
    <View style={styles.container}>
      <CameraView style={styles.camera} ref={cameraRef} />

      {/* TOP HUD */}
      <View style={[styles.hud, { backgroundColor: getDangerColor() }]}>
        {/* This text is now translated by the Python server */}
        <Text style={styles.hudStatus}>{navigationData.status}</Text>
        
        {/* This label is translated locally by the 'labels' object */}
        <Text style={styles.hudScore}>
          {labels[selectedLanguage] || "Danger"}: {navigationData.score.toFixed(2)}
        </Text>
        
        {navigationData.objects.length > 0 && (
          <View style={styles.objectContainer}>
            <Text style={styles.objectText}>
              {navigationData.objects.join(', ')}
            </Text>
          </View>
        )}
      </View>

      {/* BOTTOM LANGUAGE BUTTON */}
      <Pressable 
        style={({ pressed }) => [
          styles.langButton, 
          { opacity: pressed ? 0.7 : 1 }
        ]} 
        onPress={showLanguageMenu}
      >
        <View style={styles.langIcon}>
          <SymbolView 
            name={{ ios: 'character.bubble.fill', android: 'translate', web: 'translate' }} 
            size={20} 
            tintColor="white" 
          />
        </View>
        <Text style={styles.langButtonText}>
          {languages.find(l => l.code === selectedLanguage)?.label}
        </Text>
      </Pressable>

      {isProcessing && <View style={styles.blinkDot} />}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  camera: { flex: 1 },
  hud: {
    position: 'absolute',
    top: 60,
    left: 20,
    right: 20,
    padding: 20,
    borderRadius: 30,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.5,
    shadowRadius: 15,
    elevation: 20,
  },
  hudStatus: { color: '#fff', fontSize: 28, fontWeight: '900' },
  hudScore: { color: '#fff', fontSize: 14, marginTop: 4, opacity: 0.8 },
  objectContainer: { 
    marginTop: 12, 
    borderTopWidth: 1, 
    borderTopColor: 'rgba(255,255,255,0.2)', 
    paddingTop: 8, 
    width: '100%' 
  },
  objectText: { color: '#fff', fontSize: 13, fontWeight: '600', textAlign: 'center' },
langButton: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
    minWidth: 180, 
    height: 56,
    backgroundColor: 'rgba(0,0,0,0.85)', // Slightly darker for better contrast
    borderRadius: 28,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center', 
    paddingHorizontal: 20,
    borderWidth: 1.5,
    borderColor: 'rgba(255,255,255,0.3)',
    // Added shadow for more depth
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 5,
  },
  langButtonText: {
    color: 'white',
    fontSize: 17,
    fontWeight: '700',
    textAlign: 'center',
    // Remove marginLeft entirely to center the text perfectly
    marginLeft: 0, 
  },
  // Add this new style for the icon specifically
  langIcon: {
    position: 'absolute',
    left: 20, // Keep icon pinned to the left side
  },
  permissionText: { color: '#fff', fontSize: 18, textAlign: 'center', marginTop: '50%' },
  blinkDot: { 
    position: 'absolute', 
    bottom: 30, 
    right: 30, 
    width: 10, 
    height: 10, 
    borderRadius: 5, 
    backgroundColor: '#007AFF' 
  }
});