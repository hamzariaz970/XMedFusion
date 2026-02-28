import axios from 'axios';

// Replace with your machine's local IP address
// Use '10.0.2.2' for Android Emulators
// Use 'localhost' for iOS Simulators or Web
export const BASE_URL = 'http://192.168.1.18:8000';

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

    try {
        // We use standard fetch for streaming support on mobile
        const response = await fetch(`${BASE_URL}/api/synthesize-report`, {
            method: 'POST',
            body: formData,
            headers: {
                'Accept': 'application/x-ndjson',
            },
        });

        if (!response.ok) {
            throw new Error(`Server responded with ${response.status}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error('No reader available');

        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.trim()) {
                    try {
                        const data = JSON.parse(line) as StreamChunk;
                        onChunk(data);
                    } catch (e) {
                        console.warn('Failed to parse NDJSON line:', line);
                    }
                }
            }
        }
    } catch (error: any) {
        onChunk({ status: 'error', message: error.message || 'Network error' });
    }
};
