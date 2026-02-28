
import React from 'react';
import { View, ActivityIndicator } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { NavigationContainer, DefaultTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import {
  useFonts,
  PlusJakartaSans_400Regular,
  PlusJakartaSans_500Medium,
  PlusJakartaSans_600SemiBold,
  PlusJakartaSans_700Bold,
  PlusJakartaSans_800ExtraBold,
} from '@expo-google-fonts/plus-jakarta-sans';

// Theme
import { ThemeProvider, useTheme } from './src/theme/ThemeContext';
import { darkTheme } from './src/theme/colors';

// Screens
import SplashScreen from './src/screens/SplashScreen';
import LoginScreen from './src/screens/LoginScreen';
import UploadAnalysisScreen from './src/screens/UploadAnalysisScreen';
import MainTabNavigator from './src/navigation/MainTabNavigator';
import ReportDetailScreen from './src/screens/ReportDetailScreen';
import PatientDetailScreen from './src/screens/PatientDetailScreen';
import ProfileScreen from './src/screens/ProfileScreen';
import SecurityScreen from './src/screens/SecurityScreen';

const Stack = createNativeStackNavigator();

import { useAuth } from './src/theme/AuthContext';

function AppNavigator() {
  const { theme, isDark } = useTheme();
  const { session, loading } = useAuth();

  const navTheme = {
    ...DefaultTheme,
    colors: {
      ...DefaultTheme.colors,
      background: theme.background,
      text: theme.foreground,
      card: theme.card,
      border: theme.border,
    },
  };

  if (loading) {
    return (
      <View style={{ flex: 1, backgroundColor: theme.backgroundDeep, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator color={theme.primary} size="large" />
      </View>
    );
  }

  return (
    <NavigationContainer theme={navTheme}>
      <StatusBar style={isDark ? 'light' : 'dark'} />
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        {session ? (
          <>
            <Stack.Screen name="MainTabs" component={MainTabNavigator} />
            <Stack.Screen name="Upload" component={UploadAnalysisScreen} />
            <Stack.Screen name="ReportDetail" component={ReportDetailScreen} />
            <Stack.Screen name="PatientDetail" component={PatientDetailScreen} />
            <Stack.Screen name="Profile" component={ProfileScreen} />
            <Stack.Screen name="Security" component={SecurityScreen} />
          </>
        ) : (
          <>
            <Stack.Screen name="Login" component={LoginScreen} />
          </>
        )}
      </Stack.Navigator>
    </NavigationContainer>
  );
}

import { AuthProvider } from './src/theme/AuthContext';

export default function App() {
  const [fontsLoaded] = useFonts({
    PlusJakartaSans_400Regular,
    PlusJakartaSans_500Medium,
    PlusJakartaSans_600SemiBold,
    PlusJakartaSans_700Bold,
    PlusJakartaSans_800ExtraBold,
  });

  if (!fontsLoaded) {
    return (
      <View style={{ flex: 1, backgroundColor: darkTheme.backgroundDeep, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator color={darkTheme.primary} size="large" />
      </View>
    );
  }

  return (
    <ThemeProvider>
      <AuthProvider>
        <AppNavigator />
      </AuthProvider>
    </ThemeProvider>
  );
}
