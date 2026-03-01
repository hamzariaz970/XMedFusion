import axios from 'axios';

// Replace with your machine's local IP address, or your active ngrok URL
export const BASE_URL = 'https://prevalent-kamron-whirlingly.ngrok-free.dev';

export type StreamChunk = {
    status: string;
    message?: string;
    chunk?: string;
    final_report?: string;
    knowledge_graph?: any;
    explainability?: any;
    heatmap?: string;
    error?: string;
};

export const uploadXRay = async (
    imageUri: string,
    onChunk: (chunk: StreamChunk) => void
) => {
    const formData = new FormData();

    // Create a file object from the URI
    const filename = imageUri.split('/').pop() || 'upload.png';
    const match = /\.(\w+)$/.exec(filename);
    const type = match ? `image/${match[1]}` : `image`;

    formData.append('file', {
        uri: imageUri,
        name: filename,
        type,
    } as any);

    // React Native's fetch polyfill doesn't support streaming readable bodies (.getReader()).
    // We must use XMLHttpRequest to read chunks as they arrive.
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${BASE_URL}/api/synthesize-report`);
    xhr.setRequestHeader('Accept', 'application/x-ndjson');
    xhr.setRequestHeader('ngrok-skip-browser-warning', 'true');

    let processedIdx = 0;

    xhr.onprogress = () => {
        const text = xhr.responseText;
        const newText = text.substring(processedIdx);
        const lines = newText.split('\n');

        // If the last line is incomplete (doesn't end with \n), we leave it for the next progress event
        // by only processing up to lines.length - 1
        for (let i = 0; i < lines.length - 1; i++) {
            const line = lines[i];
            if (line.trim()) {
                try {
                    const data = JSON.parse(line) as StreamChunk;
                    onChunk(data);
                } catch (e) {
                    console.warn('Failed to parse NDJSON chunk:', line);
                }
            }
            // Update processed index strictly by the exact length of what we processed + the newline character
            processedIdx += line.length + 1;
        }
    };

    xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            // Process any remaining data that didn't end in a newline
            const remainingText = xhr.responseText.substring(processedIdx);
            if (remainingText.trim()) {
                try {
                    const data = JSON.parse(remainingText) as StreamChunk;
                    onChunk(data);
                } catch (e) {
                    // Sometimes the final empty string fails parsing, which is fine
                }
            }
        } else {
            onChunk({ status: 'error', message: `Server responded with ${xhr.status}` });
        }
    };

    xhr.onerror = () => {
        onChunk({ status: 'error', message: 'Network request failed' });
    };

    xhr.send(formData);
};
