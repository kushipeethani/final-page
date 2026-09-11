import { initializeApp, getApps, getApp } from 'firebase/app';
import { getAuth } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { getStorage } from 'firebase/storage';

export const firebaseConfig = {
  apiKey: "AIzaSyAOP8gwwW7hJupsMxcoq8chEwb4-6vtDmM",
  authDomain: "clyptus-d806a-b213d.firebaseapp.com",
  projectId: "clyptus-d806a-b213d",
  storageBucket: "clyptus-d806a-b213d.firebasestorage.app",
  messagingSenderId: "1008235920910",
  appId: "1:1008235920910:web:d99147e0a5f4bb9256065f",
  measurementId: "G-2XBYVTTZR0"
};

// Initialize Firebase
export const app = !getApps().length ? initializeApp(firebaseConfig) : getApp();
export const auth = getAuth(app);
export const db = getFirestore(app);
export const storage = getStorage(app);
export default app;
