export class AuthError extends Error {
    constructor(
        message: string,
        public status: number,
    ) {
        super(message);
        this.name = "AuthError";
    }
}

export async function authFetch(
    input: RequestInfo | URL,
    init?: RequestInit,
): Promise<Response> {
    return fetch(input, init);
}

