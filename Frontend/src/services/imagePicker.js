import { Platform } from 'react-native';

// Downscale before upload. The backend resizes to 224x224 anyway, so sending a
// full 12 MP phone photo over the LAN only costs time. 1024px keeps the image
// sharp enough that the server-side blur check still behaves sensibly.
const PICKER_OPTIONS = {
  mediaType: 'photo',
  includeBase64: false,
  maxWidth: 1024,
  maxHeight: 1024,
  quality: 0.9,
  saveToPhotos: false,
};

const PICKER_MISSING_MESSAGE =
  'The camera module is not installed yet. Run "npm install react-native-image-picker" in Frontend/, then rebuild the app.';

/**
 * react-native-image-picker is a native module, so it is required lazily:
 * that way a missing install surfaces as a readable in-app message instead of
 * a red-box crash at import time.
 */
function loadPicker() {
  try {
    // eslint-disable-next-line global-require
    return require('react-native-image-picker');
  } catch (error) {
    return null;
  }
}

function toAsset(response) {
  if (response.didCancel) {
    return { cancelled: true, asset: null };
  }

  if (response.errorCode) {
    if (response.errorCode === 'permission') {
      throw new Error(
        Platform.OS === 'android'
          ? 'Camera or storage permission was denied. Enable it in Settings → Apps → Poultry Pulse → Permissions.'
          : 'Permission was denied. Enable camera and photo access in Settings.',
      );
    }
    if (response.errorCode === 'camera_unavailable') {
      throw new Error(
        'No camera is available on this device. Use "Gallery" instead.',
      );
    }
    throw new Error(
      response.errorMessage || 'The image could not be opened.',
    );
  }

  const asset = response.assets && response.assets[0];
  if (!asset || !asset.uri) {
    throw new Error('No image was returned by the picker.');
  }

  return { cancelled: false, asset };
}

/** Open the device camera. Resolves to { cancelled, asset }. */
export async function captureFromCamera() {
  const picker = loadPicker();
  if (!picker) throw new Error(PICKER_MISSING_MESSAGE);

  const response = await picker.launchCamera(PICKER_OPTIONS);
  return toAsset(response);
}

/** Open the photo library. Resolves to { cancelled, asset }. */
export async function pickFromGallery() {
  const picker = loadPicker();
  if (!picker) throw new Error(PICKER_MISSING_MESSAGE);

  const response = await picker.launchImageLibrary({
    ...PICKER_OPTIONS,
    selectionLimit: 1,
  });
  return toAsset(response);
}
