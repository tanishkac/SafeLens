import NextAuth from "next-auth";
import type { NextAuthConfig } from "next-auth";

export const defaultUser = {
    id: "default-user",
    name: "SafeLens User",
    email: "user@safelens.local",
    image: null,
};

export const defaultSession = {
    user: defaultUser,
    expires: "2099-01-01T00:00:00.000Z",
};

export const authConfig: NextAuthConfig = {
    secret: process.env.AUTH_SECRET || "safelens-default-local-secret-32charsmin",
    providers: [],
    session: {
        strategy: "jwt",
    },
    callbacks: {
        authorized() {
            return true;
        },
        jwt({ token }) {
            token.id = defaultUser.id;
            token.name = defaultUser.name;
            token.email = defaultUser.email;
            return token;
        },
        session({ session }) {
            session.user = {
                ...session.user,
                ...defaultUser,
            };
            return session;
        },
    },
};

export const { handlers, signIn, signOut } = NextAuth(authConfig);

export async function auth() {
    return defaultSession;
}

