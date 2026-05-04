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

import { ThemeProvider, useTheme } from './src/theme/ThemeContext';
import { darkTheme } from './src/theme/colors';
import { AuthProvider, useAuth } from './src/theme/AuthContext';
import { PatientProvider } from './src/context/PatientContext';
import { AnalysisProvider } from './src/context/AnalysisContext';

import IndexScreen from './src/screens/IndexScreen';
import LoginScreen from './src/screens/LoginScreen';
import PendingApprovalScreen from './src/screens/PendingApprovalScreen';
import AdminDashboardScreen from './src/screens/AdminDashboardScreen';
import DashboardScreen from './src/screens/DashboardScreen';
import PatientsScreen from './src/screens/PatientsScreen';
import UploadAnalysisScreen from './src/screens/UploadAnalysisScreen';
import ExplainabilityScreen from './src/screens/ExplainabilityScreen';
import KnowledgeGraphScreen from './src/screens/KnowledgeGraphScreen';
import HILLabelingScreen from './src/screens/HILLabelingScreen';
import NotFoundScreen from './src/screens/NotFoundScreen';

const Stack = createNativeStackNavigator();

function AppNavigator() {
  const { theme, isDark } = useTheme();
  const { session, loading, roleLoading, isAdmin, isApproved, isPending, isRejected } = useAuth();

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

  if (loading || roleLoading) {
    return (
      <View style={{ flex: 1, backgroundColor: theme.backgroundDeep, justifyContent: 'center', alignItems: 'center' }}>
        <ActivityIndicator color={theme.primary} size="large" />
      </View>
    );
  }

  const initialRoute = !session
    ? '/'
    : isRejected || isPending || !isApproved
      ? '/pending'
      : isAdmin
        ? '/admin'
        : '/dashboard';

  return (
    <NavigationContainer theme={navTheme}>
      <StatusBar style={isDark ? 'light' : 'dark'} />
      <Stack.Navigator
        screenOptions={{ headerShown: false }}
        initialRouteName={initialRoute}
      >
        <Stack.Screen name="/" component={IndexScreen} />
        <Stack.Screen name="/login" component={LoginScreen} />
        <Stack.Screen name="/pending" component={PendingApprovalScreen} />
        <Stack.Screen name="/knowledge-graph" component={KnowledgeGraphScreen} />
        <Stack.Screen name="/admin" component={AdminDashboardScreen} />
        <Stack.Screen name="/dashboard" component={DashboardScreen} />
        <Stack.Screen name="/patients" component={PatientsScreen} />
        <Stack.Screen name="/upload" component={UploadAnalysisScreen} />
        <Stack.Screen name="/explainability" component={ExplainabilityScreen} />
        <Stack.Screen name="/hil/task/:taskId" component={HILLabelingScreen} />
        <Stack.Screen name="*" component={NotFoundScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

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
        <PatientProvider>
          <AnalysisProvider>
            <AppNavigator />
          </AnalysisProvider>
        </PatientProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}
