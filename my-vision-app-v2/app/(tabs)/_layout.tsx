import React from 'react';
import { Tabs } from 'expo-router';
import { useClientOnlyValue } from '@/components/useClientOnlyValue';

export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        // 1. This hides the top header bar (the one that says "Tab One")
        headerShown: false,
        
        // 2. This hides the bottom tab navigation bar
        tabBarStyle: { display: 'none' },

        // Keeping this for web compatibility as per your original code
        headerTransparent: useClientOnlyValue(false, true),
      }}>
      <Tabs.Screen
        name="index"
        options={{
          title: 'Home',
        }}
      />
      {/* If you aren't using the second tab at all, 
         you should delete the Tabs.Screen for "two" below 
      */}
      <Tabs.Screen
        name="two"
        options={{
          href: null, // This ensures it doesn't even occupy space in the logic
        }}
      />
    </Tabs>
  );
}