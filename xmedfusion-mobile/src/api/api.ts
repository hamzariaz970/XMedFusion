import { getApiBase } from '../lib/apiConfig';

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

const streamUpload = (
    baseUrl: string,
    imageUri: string,
    scanType: 'xray' | 'ct' | 'auto',
    onChunk: (chunk: StreamChunk) => void
) => new Promise<{ ok: boolean; status?: number; networkError?: boolean }>((resolve) => {
    const formData = new FormData();
    const filename = imageUri.split('/').pop() || 'upload.png';
    const match = /\.(\w+)$/.exec(filename);
    const extension = match?.[1]?.toLowerCase();
    const type = extension === 'jpg' ? 'image/jpeg' : extension ? `image/${extension}` : 'image';

    formData.append('files', {
        uri: imageUri,
        name: filename,
        type,
    } as any);
    formData.append('scan_type', scanType);

    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${baseUrl}/api/synthesize-report`);
    xhr.setRequestHeader('Accept', 'application/x-ndjson');
    xhr.setRequestHeader('ngrok-skip-browser-warning', 'true');

    let processedIdx = 0;

    xhr.onprogress = () => {
        const text = xhr.responseText;
        const newText = text.substring(processedIdx);
        const lines = newText.split('\n');

        for (let i = 0; i < lines.length - 1; i++) {
            const line = lines[i];
            if (line.trim()) {
                try {
                    const data = JSON.parse(line) as StreamChunk;
                    onChunk(data);
                } catch {
                    console.warn('Failed to parse NDJSON chunk:', line);
                }
            }
            processedIdx += line.length + 1;
        }
    };

    xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
            const remainingText = xhr.responseText.substring(processedIdx);
            if (remainingText.trim()) {
                try {
                    const data = JSON.parse(remainingText) as StreamChunk;
                    onChunk(data);
                } catch {
                    // Ignore incomplete final chunks.
                }
            }
            resolve({ ok: true, status: xhr.status });
        } else {
            resolve({ ok: false, status: xhr.status });
        }
    };

    xhr.onerror = () => {
        resolve({ ok: false, networkError: true });
    };

    xhr.send(formData);
});

export const uploadXRay = async (
    imageUri: string,
    scanType: 'xray' | 'ct' | 'auto' = 'xray',
    onChunk: (chunk: StreamChunk) => void
) => {
    const firstBaseUrl = await getApiBase();
    const firstAttempt = await streamUpload(firstBaseUrl, imageUri, scanType, onChunk);

    if (firstAttempt.ok) {
        return;
    }

    const retryBaseUrl = await getApiBase(true);
    if (retryBaseUrl !== firstBaseUrl) {
        const retryAttempt = await streamUpload(retryBaseUrl, imageUri, scanType, onChunk);
        if (retryAttempt.ok) {
            return;
        }

        onChunk({
            status: 'error',
            message: retryAttempt.networkError ? `Network request failed while reaching ${retryBaseUrl}` : `Server responded with ${retryAttempt.status} from ${retryBaseUrl}`,
        });
        return;
    }

    onChunk({
        status: 'error',
        message: firstAttempt.networkError ? `Network request failed while reaching ${firstBaseUrl}` : `Server responded with ${firstAttempt.status} from ${firstBaseUrl}`,
    });
};
