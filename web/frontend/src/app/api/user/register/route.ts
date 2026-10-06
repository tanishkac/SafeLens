import { NextResponse } from "next/server";
import { defaultUser } from "@/lib/auth";

export async function POST() {
    return NextResponse.json({
        success: true,
        user: defaultUser,
        isNewUser: false,
        isExistingUser: true,
    });
}

