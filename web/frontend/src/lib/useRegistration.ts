export function useRegistration() {
    return {
        registrationStatus: "idle" as const,
        error: null,
        retryRegistration: () => {},
        hideRegistration: () => {},
        needsRegistration: false,
    };
}

