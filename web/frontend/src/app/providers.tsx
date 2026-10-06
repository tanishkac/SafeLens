"use client";

import { SessionProvider } from "next-auth/react";
import { defaultSession } from "@/lib/auth";

export function Providers({ children }: { children: React.ReactNode }) {
    return (
        <SessionProvider session={defaultSession}>
            {children}
        </SessionProvider>
    );
}

