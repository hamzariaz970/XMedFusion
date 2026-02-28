import React from 'react';
import { StyleSheet, View } from 'react-native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { LayoutDashboard, Users, Network, Settings as SettingsIcon } from 'lucide-react-native';
import { useTheme } from '../theme/ThemeContext';
import { typography, fontFamily, spacing } from '../theme/colors';

import DashboardScreen from '../screens/DashboardScreen';
import PatientsScreen from '../screens/PatientsScreen';
import KnowledgeGraphScreen from '../screens/KnowledgeGraphScreen';
import SettingsScreen from '../screens/SettingsScreen';

const Tab = createBottomTabNavigator();

export default function MainTabNavigator() {
  const { theme } = useTheme();

  return (
    <Tab.Navigator
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: theme.card,
          borderTopColor: theme.border,
          borderTopWidth: 1,
          height: 68,
          paddingBottom: 10,
          paddingTop: 8,
        },
        tabBarActiveTintColor: theme.primary,
        tabBarInactiveTintColor: theme.mutedForeground,
        tabBarLabelStyle: {
          fontSize: 10,
          fontFamily: fontFamily.semiBold,
          marginTop: 4,
        },
      }}
    >
      <Tab.Screen 
        name="Dashboard" 
        component={DashboardScreen}
        options={{ 
          tabBarIcon: ({ color, size, focused }) => (
            <LayoutDashboard color={color} size={focused ? 24 : 22} strokeWidth={focused ? 2.5 : 2} />
          ) 
        }} 
      />
      <Tab.Screen 
        name="Patients" 
        component={PatientsScreen}
        options={{ 
          tabBarLabel: 'Patients', 
          tabBarIcon: ({ color, size, focused }) => (
            <Users color={color} size={focused ? 24 : 22} strokeWidth={focused ? 2.5 : 2} />
          ) 
        }} 
      />
      <Tab.Screen 
        name="KnowledgeGraph" 
        component={KnowledgeGraphScreen}
        options={{ 
          tabBarLabel: 'Graph', 
          tabBarIcon: ({ color, size, focused }) => (
            <Network color={color} size={focused ? 24 : 22} strokeWidth={focused ? 2.5 : 2} />
          ) 
        }} 
      />
      <Tab.Screen 
        name="Admin" 
        component={SettingsScreen}
        options={{ 
          tabBarLabel: 'Admin', 
          tabBarIcon: ({ color, size, focused }) => (
            <SettingsIcon color={color} size={focused ? 24 : 22} strokeWidth={focused ? 2.5 : 2} />
          ) 
        }} 
      />
    </Tab.Navigator>
  );
}
