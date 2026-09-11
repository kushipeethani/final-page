import { ref, uploadBytesResumable, getDownloadURL } from 'firebase/storage';
import { storage } from '../config/firebase';

export interface FirebaseUploadResult {
  fileName: string;
  downloadUrl: string;
  storagePath: string;
}

/**
 * Upload a resume file to Firebase Cloud Storage and get its public download URL
 */
export const uploadResumeToFirebase = async (
  file: File,
  userId?: string | number,
  onProgress?: (progress: number) => void
): Promise<FirebaseUploadResult> => {
  const timestamp = Date.now();
  const safeName = file.name.replace(/[^a-zA-Z0-9.-]/g, '_');
  const userFolder = userId ? `users/${userId}` : 'public';
  const storagePath = `resumes/${userFolder}/${timestamp}_${safeName}`;
  const storageRef = ref(storage, storagePath);

  const uploadTask = uploadBytesResumable(storageRef, file);

  return new Promise((resolve, reject) => {
    uploadTask.on(
      'state_changed',
      (snapshot) => {
        if (onProgress && snapshot.totalBytes > 0) {
          const progress = (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
          onProgress(progress);
        }
      },
      (error) => {
        reject(error);
      },
      async () => {
        try {
          const downloadUrl = await getDownloadURL(uploadTask.snapshot.ref);
          resolve({
            fileName: file.name,
            downloadUrl,
            storagePath,
          });
        } catch (err) {
          reject(err);
        }
      }
    );
  });
};
