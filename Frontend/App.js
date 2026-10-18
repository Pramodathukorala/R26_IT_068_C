import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import { PaperProvider } from 'react-native-paper';
import { SafeAreaProvider } from 'react-native-safe-area-context';

import DashboardScreen from './src/screens/DashboardScreen';
import ChickenBehaviourModule from './src/screens/ChickenBehaviourModule';
import DiseaseDiagnosticScreen from './src/screens/DiseaseDiagnosticScreen';
import PredatorDetectionScreen from './src/screens/PredatorDetectionScreen';
import PredatorAlarmProvider from './src/components/PredatorAlarmProvider';

const Stack = createNativeStackNavigator();

export default function App() {
  return (
    <SafeAreaProvider>
      <PaperProvider>
        <NavigationContainer>
          {/* Wraps the navigator so the predator siren keeps watching on every
              screen, not just the Predator Detection route. */}
          <PredatorAlarmProvider>
            <Stack.Navigator
              screenOptions={{ headerShown: false, animation: 'slide_from_right' }}
              initialRouteName="Dashboard"
            >
              <Stack.Screen name="Dashboard" component={DashboardScreen} />
              <Stack.Screen name="ChickenBehaviour" component={ChickenBehaviourModule} />
              <Stack.Screen name="DiseaseDetection" component={DiseaseDiagnosticScreen} />
              <Stack.Screen name="PredatorDetection" component={PredatorDetectionScreen} />
            </Stack.Navigator>
          </PredatorAlarmProvider>
        </NavigationContainer>
      </PaperProvider>
    </SafeAreaProvider>
  );
}
